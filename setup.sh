#!/bin/bash
# Quick Setup and Run Script for Toxicity Prediction Project

echo "🧬 Toxicity Prediction - Setup and Run Script"
echo "=============================================="
echo ""

# Check Python installation
if ! command -v python &> /dev/null; then
    echo "❌ Python is not installed. Please install Python 3.8 or higher."
    exit 1
fi

python_version=$(python --version)
echo "✅ Using: $python_version"
echo ""

# Create virtual environment (optional)
read -p "Create a new virtual environment? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Creating virtual environment..."
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    echo "✅ Virtual environment created and activated"
fi

echo ""
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✅ Dependencies installed successfully"
else
    echo "❌ Failed to install dependencies"
    exit 1
fi

echo ""
echo "=============================================="
echo "Setup Complete! Choose how to run:"
echo "=============================================="
echo ""
echo "1) Run Streamlit Web App (Recommended)"
echo "   Command: streamlit run app.py"
echo ""
echo "2) Run Training Script"
echo "   Command: python src/train.py"
echo ""
echo "3) Run Jupyter Notebooks"
echo "   Command: jupyter lab notebooks/"
echo ""
echo "4) Install Package in Development Mode"
echo "   Command: pip install -e ."
echo ""

read -p "Select option (1-4): " option

case $option in
    1)
        echo "Launching Streamlit app..."
        streamlit run app.py
        ;;
    2)
        echo "Starting training..."
        python src/train.py --config src/config.yaml
        ;;
    3)
        echo "Launching Jupyter Lab..."
        jupyter lab notebooks/
        ;;
    4)
        echo "Installing package in development mode..."
        pip install -e .
        echo "✅ Package installed. You can now import: from toxicity import ..."
        ;;
    *)
        echo "Invalid option"
        exit 1
        ;;
esac
