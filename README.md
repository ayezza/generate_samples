# 📊 Statistical Sample Generator and Analyzer

This project aims to :

1. Demonstrate Large Number Law by generating and analyzing statistical samples from normal distributions

2. Provide tools for generating and analyzing data, including visualization and regression analysis

3. Accepts parameters for the population size, the number of samples, the number of graphs to generate, the degree of the polynomial to use for regression, and the folders to save output CSV files and graphs in

## ✨ Features

- Generate random population samples with configurable sizes
- Analyze sample distributions with various statistical metrics
- Generate visualizations including:
  - Individual population analysis (4 graphs per population)
  - Combined population analysis
  - Sample distribution histograms
  - Box plots
  - Normal distribution fitting
  - Linear and polynomial regression
- Export results to CSV files
- Statistical analysis including:
  - Mean, median, mode
  - Standard deviation
  - Distribution parameters
  - Confidence intervals

## 📦 Installation

1. **Create a virtual environment**:
```bash
python -m venv venv
```

2. **Activate the virtual environment**:
- **Windows**:
```bash
venv\Scripts\activate
```
- **Unix/MacOS**:
```bash
source venv/bin/activate
```

3. **Install dependencies**:
```bash
pip install numpy pandas scipy scikit-learn matplotlib
```

## 📚 Usage

### 📚 Basic Usage
```bash
python main.py [population_size] [csv_output_file] [generate_graphs] [polynomial_degree] [graphs_folder] [show_graphs] [generate_combined_graph]
```

### 📚 Parameters

- `population_size`: Size of population (default: 1000)
- `csv_output_file`: Path to output CSV file (default: './output.csv')
- `generate_graphs`: Generate graphs (1) or not (0) (default: 1)
- `polynomial_degree`: Degree for polynomial regression (default: 2)
- `graphs_folder`: Folder for saving graphs (default: './graphs/')
- `show_graphs`: Display graphs (1) or not (0) (default: 1)
- `generate_combined_graph`: Graph generation mode:
  - 1: Generate separate graphs only (default)
  - 2: Generate combined graph only
  - 3: Generate both separate and combined graphs

### 📚 Examples

1. **Default parameters**:
```bash
python main.py
```

2. **Custom population with separate graphs**:
```bash
python main.py 10000 "output_10000.csv" 1 2 "./graphs" 0 1
```

3. **Custom population with combined graph only**:
```bash
python main.py 10000 "output_10000.csv" 1 2 "./graphs" 0 2
```

4. **Custom population with both graph types**:
```bash
python main.py 10000 "output_10000.csv" 1 2 "./graphs" 0 3
```

5. **Run tests**:
```bash
python main.py --test
```

## 📄 Output

### 📄 Generated Files

1. **CSV file containing**:
   - Sample sizes
   - Population statistics
   - Mean values
   - Other statistical metrics

2. **Graphs (if enabled)**:
   - Individual population analysis (sample_size_XXX.png)
   - Combined population analysis (combined_populations.png)
   - Normal distribution fits
   - Box plots
   - Regression analysis plots

### 📊 Visualizations

The program generates several types of plots depending on the `generate_combined_graph` parameter:
- Individual population analysis (4 graphs per population)
  - Distribution analysis with μ, σ, and mode indicators
  - Box plots showing quartiles and outliers
  - Histograms and normal distribution fits
- Combined population analysis showing all populations in one graph
- Regression analysis for sample means

## 🧪 Testing

Run the automated test suite:
```bash
python main.py --test
```

This will execute various test scenarios and verify:
- Basic functionality
- Large population handling
- Graph generation options
- File output handling

## 📜 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 🤝 Regards

Abdel YEZZA, Ph.D