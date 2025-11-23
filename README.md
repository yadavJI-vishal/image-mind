📌**Project Overview**

ImageMind is a **Flask-based Visual Question Answering (VQA)** system that uses deep learning to understand images and answer natural language questions about them.
- Upload any image 📷
- Ask a question about it ❓
- Get top AI-powered answers 🎯
The system combines **ResNet50** for image feature extraction with a **language model** to process questions and predict answers.

**🚀 Features**
- 🖼️ Image understanding via pre-trained ResNet50 CNN
- 💬 Question processing with tokenization and sequence padding
- 🧠 Custom trained VQA model to combine image and question features
- 🌐 Interactive Flask web interface for uploading images and submitting questions
- 🔥 Real-time predictions with top 5 answers and confidence scores

**🗂️ Project Structure**

ImageMind/
├── app.py # Flask backend and prediction logic
├── static/ # Static files (uploads, CSS, etc.)
├── templates/ # HTML templates for Flask UI
├── vqa_model_50k.h5 # Trained VQA model (use Git LFS for large file)
├── question_tokenizer.pkl # Tokenizer for questions
├── answer_mappings.pkl # Answer <-> index mappings
└── README.md # Project documentation



**🛠️ Installation & Setup**
**1️⃣ Clone the repository**
      git clone https://github.com/your-username/image-mind.git
      cd image-mind
**2️⃣ Install dependencies**
      python -m venv venv
**3️⃣ Run the Flask app**
      python app.py

**📸 How to Use**
- Click Upload Image and select your image file (PNG, JPG, JPEG).
- Type your question in the text box below.
- Press Submit to get answers.

**🧑‍💻 Technologies Used**
| Component                | Technology                   |
| ------------------------ | ---------------------------- |
| Image Feature Extraction | ResNet50 CNN                 |
| Language Processing      | Keras Tokenizer + Padding    |
| Model Architecture       | CNN + LSTM Fusion            |
| Backend                  | Flask Web Framework          |
| Deployment               | Localhost / Cloud (optional) |

**🤝 Contributing**
Contributions, issues, and feature requests are welcome!

**📧 Contact**
Vishal Yadav
Email: vy5068@gmail.com
GitHub: https://github.com/yadavJI-vishal

Feel free to fork the repo and submit a pull request.
