const imageInput = document.getElementById("image-input");
const dropZone = document.getElementById("drop-zone");
const fileName = document.getElementById("file-name");
const previewWrap = document.getElementById("preview-wrap");
const previewImage = document.getElementById("preview-image");
const analyzeButton = document.getElementById("analyze-button");
const loadingMessage = document.getElementById("loading-message");
const errorMessage = document.getElementById("error-message");
const resultsSection = document.getElementById("results-section");
const originalImage = document.getElementById("original-image");
const resultImage = document.getElementById("result-image");
const detectionsList = document.getElementById("detections-list");
const resultCount = document.getElementById("result-count");

let selectedFile = null;

function showError(message) {
    errorMessage.textContent = message;
    errorMessage.classList.remove("hidden");
}

function clearError() {
    errorMessage.textContent = "";
    errorMessage.classList.add("hidden");
}

function chooseFile(file) {
    clearError();
    if (!file) return;
    const validTypes = ["image/jpeg", "image/png"];
    if (!validTypes.includes(file.type)) {
        selectedFile = null;
        analyzeButton.disabled = true;
        showError("Please choose a JPG, JPEG, or PNG image.");
        return;
    }
    if (file.size > 10 * 1024 * 1024) {
        selectedFile = null;
        analyzeButton.disabled = true;
        showError("The image must be smaller than 10 MB.");
        return;
    }

    selectedFile = file;
    fileName.textContent = file.name;
    previewImage.src = URL.createObjectURL(file);
    previewWrap.classList.remove("hidden");
    analyzeButton.disabled = false;
    resultsSection.classList.add("hidden");
}

imageInput.addEventListener("change", (event) => chooseFile(event.target.files[0]));
["dragenter", "dragover"].forEach((eventName) => dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.add("dragging");
}));
["dragleave", "drop"].forEach((eventName) => dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.remove("dragging");
}));
dropZone.addEventListener("drop", (event) => chooseFile(event.dataTransfer.files[0]));

function renderDetections(detections) {
    resultCount.textContent = `${detections.length} detection${detections.length === 1 ? "" : "s"}`;
    if (!detections.length) {
        detectionsList.innerHTML = '<div class="empty-state">No tumor detection was returned by the model.</div>';
        return;
    }

    detectionsList.innerHTML = detections.map((detection) => {
        const box = detection.box;
        return `<div class="detection-row">
            <div><div class="detection-name">${escapeHtml(detection.class_name)}</div>
            <div class="detection-box">Box: (${box.x1}, ${box.y1}) to (${box.x2}, ${box.y2})</div></div>
            <div class="detection-confidence">${detection.confidence_percent.toFixed(2)}%</div>
        </div>`;
    }).join("");
}

function escapeHtml(value) {
    const element = document.createElement("div");
    element.textContent = value;
    return element.innerHTML;
}

analyzeButton.addEventListener("click", async () => {
    if (!selectedFile) {
        showError("Please select an MRI image first.");
        return;
    }

    clearError();
    analyzeButton.disabled = true;
    loadingMessage.classList.remove("hidden");
    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
        const response = await fetch("/predict", { method: "POST", body: formData });
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "The server could not analyze this image.");

        originalImage.src = previewImage.src;
        resultImage.src = `${data.result_url}?t=${Date.now()}`;
        renderDetections(data.detections);
        resultsSection.classList.remove("hidden");
        resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (error) {
        showError(error.message || "Something went wrong while analyzing the image.");
    } finally {
        analyzeButton.disabled = false;
        loadingMessage.classList.add("hidden");
    }
});
