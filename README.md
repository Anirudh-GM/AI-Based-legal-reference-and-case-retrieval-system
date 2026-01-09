# AI-Based Legal Reference and Case Retrieval System

A comprehensive legal reference system that leverages AI to help users find relevant legal cases and references quickly and efficiently.

## 🚀 Features

- **Legal Document Search**: Search through a database of legal documents and cases
- **AI-Powered Retrieval**: Advanced search capabilities using AI/ML algorithms
- **User Authentication**: Secure user accounts with profile management
- **Milestone-Based Structure**: Organized development progress across multiple milestones

## 📁 Project Structure

```
.
├── Milestone-1/          # Initial project setup and basic functionality
│   ├── task1.py
│   ├── task2.py
│   └── task3.py
│
├── Milestone-2/          # Core system implementation
│   ├── system_template.py
│   ├── task4.py
│   ├── task5.py
│   ├── task6.py
│   ├── task7.py
│   └── task8.py
│
├── Milestone-3/          # Web application and user interface
│   ├── app.py            # Main application file
│   ├── upload.py         # File upload handler
│   ├── system_template.py
│   ├── static/           # Static files (CSS, JS, images)
│   └── templates/        # HTML templates
│
├── .gitignore           # Specifies intentionally untracked files to ignore
└── README.md            # This file
```

## 🛠️ Setup and Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Anirudh-GM/AI-Based-legal-reference-and-case-retrieval-system.git
   cd AI-Based-legal-reference-and-case-retrieval-system
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   Create a `.env` file in the root directory with the required variables:
   ```
   FLASK_APP=Milestone-3/app.py
   FLASK_ENV=development
   SECRET_KEY=your-secret-key-here
   ```

5. **Run the application**
   ```bash
   cd Milestone-3
   flask run
   ```

   The application will be available at `http://localhost:5000`

## 📝 Usage

1. Access the web interface at `http://localhost:5000`
2. Create an account or log in
3. Upload legal documents or search the existing database
4. Use the search functionality to find relevant legal cases and references

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Developed by [Your Name] | [GitHub Profile](https://github.com/Anirudh-GM)
