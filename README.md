# Device Recommendation LLM Fine-Tuning Project (LoRa fine tuned for reasoning)

A lightweight AI-powered device recommendation system built using **Llama 3.2 3B Instruct**, **QLoRA**, **PEFT**, and **TRL**.

This project fine-tunes a large language model to act as an intelligent “device expert” that recommends devices based on user needs, budget, profession, and usage patterns.

---

# 🚀 Features

- Fine-tuning using LoRA (Low Rank Adaptation)
- 4-bit quantization using BitsAndBytes
- Memory efficient training
- Uses Meta Llama 3.2 3B Instruct
- Custom instruction + reasoning dataset
- Generates human-like recommendations
- Upload trained adapters directly to Hugging Face

---

# 🧠 Tech Stack

- Python
- PyTorch
- Transformers
- TRL
- PEFT
- BitsAndBytes
- Hugging Face Hub
- Google Colab

---

# 📁 Project Structure

```bash
device-recommender-llm/
│
├── dataset/
│   └── master_device_reasoning_dataset.jsonl
│
├── checkpoints/
│   └── checkpoint-300/
│
├── notebooks/
│   └── training.ipynb
│
├── inference/
│   └── inference.py
│
├── requirements.txt
│
└── README.md
```

---

# 📦 Installation

Clone the repository:

```bash
git clone https://github.com/your-username/device-recommender-llm.git
cd device-recommender-llm
```

Install dependencies:

```bash
pip install -U bitsandbytes transformers peft accelerate trl datasets
pip install --upgrade xformers
```

---

# 🔥 Model Used

```python
meta-llama/Llama-3.2-3B-Instruct
```

This model is loaded in 4-bit quantized mode for efficient training.

---

# ⚡ Quantization Configuration

```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)
```

---

# 🧩 LoRA Configuration

```python
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
```

---

# 📚 Dataset Format

The dataset is stored in `.jsonl` format.

Example:

```json
{
  "instruction": "Suggest a laptop for AI development",
  "reasoning": "The user needs GPU acceleration and enough RAM",
  "output": "A RTX 4060 laptop with 16GB RAM would be suitable"
}
```

---

# 🛠 Training

```python
trainer.train()
```

Training uses:

- Gradient Accumulation
- QLoRA
- PEFT
- TRL SFTTrainer

---

# 📈 Training Configuration

```python
training_args = SFTConfig(
    output_dir="llama3_no_unsloth",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    max_steps=300,
    optim="paged_adamw_8bit"
)
```

---

# 🤖 Inference Example

```python
user_input = "I want to buy a device for AI powered features"

ask_the_expert(user_input)
```

Example Output:

```text
Based on your needs, a mid-range AI laptop with
good GPU acceleration and battery backup would
be a better choice.
```

---

# ☁️ Upload to Hugging Face

```python
api.upload_folder(
    folder_path="checkpoint-300",
    repo_id="hawkiee/Device_recommender",
    repo_type="model"
)
```

---

# 🧪 Future Improvements

- Add RAG support
- Add memory-based recommendations
- Create Streamlit or React frontend
- Add multilingual support
- Add voice assistant integration
- Deploy with FastAPI + Docker

---

# 📌 Applications

- AI shopping assistants
- Smart device recommendation engines
- E-commerce AI agents
- Personalized gadget advisors
- Chat-based product assistants

---

# 🙌 Acknowledgements

- Hugging Face
- Meta AI
- Transformers Library
- TRL Library
- PEFT Library

---

# 👨‍💻 Author

Developed as an experimental LLM fine-tuning project for intelligent device recommendation systems.
