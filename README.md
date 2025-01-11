# Statistical Sample Generator and Analyzer

This project provides tools for generating and analyzing statistical samples from normal distributions, including visualization and regression analysis.

## Features

- Generate random population samples with configurable sizes
- Analyze sample distributions with various statistical metrics
- Generate visualizations including:
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

## Installation

1. **Create a virtual environment**:

```bash
venv\Scripts\activate
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

## Usage

### Basic Usage
```bash
python main.py [population_size] [csv_output_file] [generate_graphs] [polynomial_degree] [show_graphs] [graphs_folder]
```


### Parameters

- `population_size`: Size of population (default: 1000)
- `csv_output_file`: Path to output CSV file (default: './output.csv')
- `generate_graphs`: Generate graphs (1) or not (0) (default: 1)
- `polynomial_degree`: Degree for polynomial regression (default: 2)
- `show_graphs`: Display graphs (1) or not (0) (default: 1)
- `graphs_folder`: Folder for saving graphs (default: './graphs/')

### Examples

1. **Default parameters**:
```bash
python main.py
```

2. **Custom population size and output**:
```bash
python main.py 10000 "output_10000.csv" 1 2 0 "./graphs"
```

3. **Run tests**: (you have to close the interactive graphs)
```bash
python main.py --test
```

## Output

### Generated Files

1. **CSV file containing**:
   - Sample sizes
   - Population statistics
   - Mean values
   - Other statistical metrics

2. **Graphs (if enabled)**:
   - Sample distribution histograms
   - Normal distribution fits
   - Box plots
   - Regression analysis plots

### Visualizations

The program generates several types of plots:
- Distribution analysis with μ, σ, and mode indicators
- Box plots showing quartiles and outliers
- Regression analysis for sample means
- Combined statistical visualizations

## Testing

**Run the automated test suite**:
```bash
python main.py --test
```

**This will execute various test scenarios and verify**:
- Basic functionality
- Large population handling
- Graph generation
- File output handling

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.