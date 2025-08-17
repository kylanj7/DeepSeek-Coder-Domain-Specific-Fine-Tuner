# DeepSeek Coder Domain-Specific Fine-Tuner

A comprehensive pipeline for fine-tuning DeepSeek Coder models on domain-specific programming content extracted from PDF documents. This system uses Parameter-Efficient Fine-Tuning (PEFT) with LoRA adapters to create specialized coding assistants optimized for RTX 3090 GPUs.

## 📁 Project Structure

```
deepseek-finetuner/
├── MainPipeline.py         # Main orchestration script
├── PDFprocessing.py        # PDF extraction and preprocessing
├── FineTuning.py          # LoRA fine-tuning implementation  
├── ModelTesting.py        # Model evaluation and testing
├── requirements.txt       # Python dependencies
├── pdfs/                  # Place your PDF documents here
├── processed_data/        # Generated training data
│   └── training_data.json # Processed instruction pairs
├── models/               # Saved fine-tuned models
└── deepseek-coder-finetuned/  # Output directory for trained model
    ├── adapter_config.json
    ├── adapter_model.bin
    └── tokenizer files
```

## 🚀 Features

- **PDF-to-Training Pipeline**: Automatically extract and process programming content from PDF documents
- **Memory-Efficient Fine-Tuning**: Uses LoRA (Low-Rank Adaptation) for efficient training on consumer GPUs
- **RTX 3090 Optimized**: Specifically tuned for 24GB VRAM constraints with smart memory management
- **Instruction Format**: Converts content into proper instruction-following format for the DeepSeek Coder model
- **Interactive Testing**: Built-in testing interface to evaluate your fine-tuned model
- **Modular Architecture**: Cleanly separated components for processing, training, and testing

## 📋 Requirements

### Hardware
- NVIDIA GPU with CUDA support (optimized for RTX 3090 24GB)
- Minimum 16GB system RAM
- 50GB+ free disk space

### Software
- Python 3.8+
- CUDA 11.8+ or 12.x
- Git

## 🛠️ Installation

1. **Clone and setup environment:**
```bash
git clone <your-repo-url>
cd deepseek-finetuner
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Verify GPU setup:**
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## 🔧 Core Components

## 🚦 Quick Start

1. **Add your PDF documents:**
```bash
mkdir -p pdfs
# Copy your programming PDFs to the pdfs/ directory
```

2. **Run the complete pipeline:**
```bash
python MainPipeline.py --step all
```

Or run individual steps:
```bash
# Process PDFs only
python MainPipeline.py --step process

# Fine-tune model only  
python MainPipeline.py --step train

# Test model only
python MainPipeline.py --step test
```

3. **Test your fine-tuned model:**
```bash
python ModelTesting.py
```

## 🔧 Core Components

### PDF Processing (`PDFprocessing.py`)
Handles extraction and preprocessing of programming content from PDFs:
- **Text Extraction**: Uses PyMuPDF for high-quality text extraction with formatting preservation
- **Content Cleaning**: Removes PDF artifacts, page headers, and formatting issues
- **Section Splitting**: Intelligently divides content into logical sections
- **Instruction Generation**: Converts sections into instruction-response pairs

### Fine-Tuning (`FineTuning.py`)
Implements efficient LoRA fine-tuning:
- **Model Loading**: Loads DeepSeek Coder with 8-bit quantization for memory efficiency
- **LoRA Configuration**: Applies low-rank adapters to attention layers
- **Training Loop**: Handles the complete training process with gradient accumulation
- **Memory Optimization**: Configured for RTX 3090 constraints

Key LoRA settings:
```python
lora_config = LoraConfig(
    r=16,                    # Rank
    lora_alpha=32,          # Scaling factor
    lora_dropout=0.1,       # Dropout rate
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)
```

### Model Testing (`ModelTesting.py`)
Comprehensive evaluation framework:
- **Response Generation**: Generate responses to custom instructions
- **Interactive Mode**: Real-time testing interface
- **Predefined Tests**: Standard evaluation scenarios
- **Performance Monitoring**: Track model quality and response time

## ⚙️ Configuration

### Training Parameters
Key parameters in `FineTuning.py`:
```python
training_args = TrainingArguments(
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    fp16=True,
    eval_steps=100,
    save_steps=500
)
```

### Memory Optimization
For different GPU configurations:
- **RTX 3090 (24GB)**: Default settings work well
- **RTX 3080 (10GB)**: Reduce `per_device_train_batch_size` to 1, increase `gradient_accumulation_steps`
- **RTX 4090 (24GB)**: Can increase batch size or reduce quantization

## 📊 Expected Performance

### Training Time
- **RTX 3090**: ~2-4 hours for 1000 instruction pairs
- **Memory Usage**: ~20GB VRAM during training
- **Disk Space**: ~15GB for base model + adapters

### Model Quality
The fine-tuned model will:
- Understand domain-specific terminology from your PDFs
- Generate responses in the style of your training content
- Maintain general programming knowledge from the base model
- Follow instruction-response format consistently

## 🔍 Monitoring and Debugging

### GPU Monitoring
```bash
# Watch GPU usage during training
watch -n 1 nvidia-smi
```

### Training Logs
The system uses Weights & Biases for experiment tracking:
- Loss curves
- Learning rate schedules  
- GPU utilization
- Training metrics

### Common Issues

**Out of Memory Error:**
- Reduce `per_device_train_batch_size`
- Increase `gradient_accumulation_steps`
- Enable `load_in_8bit=True`

**No Trainable Parameters:**
- Verify LoRA configuration
- Check target module names for your model
- Ensure `requires_grad=True` for LoRA parameters

**Poor PDF Extraction:**
- Check PDF quality and text layers
- Adjust regex patterns in `clean_pdf_artifacts()`
- Manually verify extracted content

## 🧪 Testing Your Model

### Interactive Testing
```bash
python ModelTesting.py
# Choose option 2 for interactive mode
```

### Programmatic Testing
```python
from ModelTesting import ModelTester

tester = ModelTester()
tester.load_model()

response = tester.generate_response(
    instruction="Explain Python decorators",
    input_text=""
)
print(response)
```

## 📈 Advanced Usage

### Custom Data Format
Modify `PDFprocessing.py` to handle your specific PDF structure:
```python
def create_custom_pairs(self, text: str) -> List[Dict]:
    # Implement custom logic for your domain
    pass
```

### Hyperparameter Tuning
Experiment with different LoRA configurations:
- Increase `r` for more parameters (higher quality, more memory)
- Adjust `lora_alpha` for learning rate scaling
- Modify `target_modules` for different attention patterns

### Multi-GPU Training
Enable distributed training for multiple GPUs:
```python
# In training_args
local_rank=int(os.environ.get("LOCAL_RANK", -1))
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **DeepSeek AI** for the excellent base models
- **Hugging Face** for transformers and PEFT libraries
- **Microsoft** for the LoRA technique
- **PyMuPDF** for robust PDF processing

## 📞 Support

- Create an issue for bugs or feature requests
- Check the troubleshooting section for common problems
- Review the code comments for implementation details

---

**Note**: This fine-tuner is designed for educational and research purposes. Ensure you have proper licensing for any PDF content you use for training.
