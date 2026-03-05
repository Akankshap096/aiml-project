### **PlantCare AI**

Intelligent Plant Disease Detection using Transfer Learning.

#### **Overview**

PlantCare AI is an intelligent deep learning system designed to identify plant diseases from leaf images.
Using Transfer Learning with MobileNetV2, the system can analyze plant leaves and predict the disease class with high accuracy.
Early detection of plant diseases is crucial for improving crop productivity and preventing agricultural losses. PlantCare AI aims to provide a simple and accessible AI-powered tool that helps farmers, researchers, and gardeners detect plant diseases quickly.
This project demonstrates how Artificial Intelligence and Computer Vision can be applied to solve real-world agricultural problems.

#### **Project Goals**

The main objectives of PlantCare AI are:

• Detect plant diseases automatically from leaf images

• Apply transfer learning to build an efficient deep learning model

• Provide a simple interface where users can upload leaf images

• Reduce the time required for manual disease inspection

• Demonstrate real-world application of AI in agriculture

#### **AI Model \& Approach :**

PlantCare AI uses a Convolutional Neural Network (CNN) with transfer learning.

**Model Used :**

MobileNetV2

Why MobileNetV2?

✔ Lightweight architecture

✔ High accuracy in image classification

✔ Efficient for real-time applications

✔ Pretrained on ImageNet dataset

**Model Pipeline :**

Dataset → Image Preprocessing → Feature Extraction → Transfer Learning → Disease Classification

#### **Technology Stack :**

1.Programming :
Python 

2.Deep Learning :
TensorFlow
Keras
MobileNetV2

3.Libraries :
NumPy
OpenCV
Matplotlib
Scikit-learn
Pillow

4.Application Layer :
Flask – Backend API

HTML / CSS / JavaScript – User Interface\

#### Supported Crops & Disease Classes (38)

| Crop | Diseases |

| Apple | Apple Scab, Black Rot, Cedar Apple Rust, Healthy |

| Corn | Gray Leaf Spot, Common Rust, Northern Leaf Blight, Healthy |

| Grape | Black Rot, Esca, Leaf Blight, Healthy |

| Tomato | Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria, Spider Mites, Target Spot, Yellow Leaf Curl, Mosaic Virus, Healthy |

| Potato | Early Blight, Late Blight, Healthy |

| Pepper | Bacterial Spot, Healthy |

| Peach | Bacterial Spot, Healthy |

| Strawberry | Leaf Scorch, Healthy |

| Cherry | Powdery Mildew, Healthy |

| Blueberry | Healthy |

| Orange | Citrus Greening (HLB) |

| Soybean | Healthy |

| Squash | Powdery Mildew |

| Raspberry | Healthy |

#### **System Architecture**

User Uploads Leaf Image

↓

Image Preprocessing

↓

Deep Learning Model (MobileNetV2)

↓

Disease Prediction

↓

Result Display

#### **How to Run the Project**

1️⃣ Clone the Repository
git clone https://github.com/Akankshap096/aiml-project

2️⃣ Install Required Libraries
pip install -r requirements.txt

3️⃣ Run the Application
python backend/app.py

4️⃣ Use the System
Upload a plant leaf image and the system will predict the disease.

#### **Team Members :**

Akansha patel 

Ragni patel 

Nikhil tirgude 

Kumar Harshwardhan  

#### **Real-World Impact**

PlantCare AI demonstrates how Artificial Intelligence can contribute to smart agriculture by enabling:

• Early disease detection

• Reduced crop losses

• Faster diagnosis

• Technology-driven farming solutions

##  Future Plans

- Mobile app for field use
- Real-time farm IoT integration
- Educational platform integration
- Expanded crop and disease coverage
- Cloud deployment (AWS / GCP)




