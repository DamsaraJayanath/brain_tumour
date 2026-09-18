# Brain Tumor Detection using YOLO and FastAPI

This is a computer vision project I built to detect brain tumors in MRI images using YOLO.

I trained a YOLO model using a brain tumor dataset from Roboflow and then built a simple FastAPI web application to use the trained model.

## What the project does

You can upload an MRI image through the web page.

The application then:

1. Takes the uploaded MRI image
2. Sends it to the FastAPI backend
3. Runs the YOLO model
4. Detects the tumor
5. Draws the detection on the image
6. Shows the result and confidence score

## Technologies Used

* Python
* YOLO
* Ultralytics
* OpenCV
* FastAPI
* HTML
* CSS
* JavaScript

## Project Structure

```text
brain-tumor-detection/
│
├── best.pt
├── main.py
├── requirements.txt
│
├── templates/
│   └── index.html
│
├── static/
│   ├── style.css
│   └── script.js
│
├── uploads/
└── results/
```

## How to Run

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd brain_tumor
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Make sure `best.pt` is in the project folder.

Then start FastAPI:

```bash
uvicorn main:app --reload
```

Open this in your browser:

```text
http://127.0.0.1:8000
```

## Model

The trained YOLO model is saved as:

```text
best.pt
```

The model was trained to detect these classes:

* Glioma
* Meningioma
* No Tumour
* Pituitary

## Dataset

The dataset used for this project is the **Brain Tumour Computer Vision Dataset** created by Shubham on Roboflow Universe.

Dataset:
https://universe.roboflow.com/shubham-qvz5f/brain-tumour-kajng

The dataset is licensed under **CC BY 4.0**.

Some sample MRI images included in this repository are from this dataset.

## Disclaimer

This project is made for learning and research purposes.

It is not a medical diagnostic tool and should not be used for medical decisions.

## Project Demo

![image alt](https://github.com/DamsaraJayanath/brain_tumour/blob/afbe6a4a586978839315fdddf6f36b6a5c06c645/SS1.png)

### Detection Result

![image alt](https://github.com/DamsaraJayanath/brain_tumour/blob/9b93a73205e9a1c5c324843263a62cba86e00104/result.png)
