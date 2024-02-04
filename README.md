# Meteor Data Insights Dashboard

## Overview

Explore and visualize meteorite data with the Meteor Data Insights Dashboard. This web application is built using Dash and Plotly, offering various interactive visualizations to gain insights into meteorite falls worldwide.

## Features

### Visualizations

1. **World Map of Meteor Mass:**
   - Displays meteorite landings worldwide.
   - The size of markers corresponds to the mass of meteorites.
   - Color indicates whether the meteorite fell or was found.

2. **Meteor Falls Over the Years:**
   - Bar chart displaying the number of meteor falls each year.

3. **Fall Distribution:**
   - Pie chart illustrating the distribution of meteor falls.

4. **Top 50 Meteor Classes:**
   - Bar chart showcasing the top 50 meteor classes by count.

5. **Meteor Counts Over the Decade:**
   - Bar chart illustrating the count of meteor falls aggregated by decade.

6. **Mass Distribution by Meteorite Class:**
   - Box plot displaying the distribution of meteorite masses by class.

7. **Correlation Heatmap:**
   - Heatmap showing the correlation between different numerical features in the dataset.

8. **Latitude Distribution:**
   - Scatter plot displaying the distribution of meteorites based on latitude.

9. **Longitude Distribution:**
    - Scatter plot displaying the distribution of meteorites based on longitude.

10. **Top 50 Meteor Masses Over the Years:**
    - Line chart showcasing the top 50 meteor masses over the years.
    - Each data point is marked, and they are arranged in ascending order of the year.

11. **Meteorite Mass Statistics and Distribution by Class:**
    - Subplots displaying statistics (mean, median, standard deviation) of meteorite masses by class.
    - Includes a box plot showing the distribution of meteorite masses by class.



## Getting Started

### Prerequisites

- Python (version 3.7 or higher)
- Pip (to install dependencies)

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Aatisha/meteor-viz.git
   cd meteor-viz
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

### Usage

Run the following command to start the dashboard:

```bash
python meteor_app.py
```

Visit [http://127.0.0.1:8050/](http://127.0.0.1:8050/) in your web browser to access the Meteor Data Insights Dashboard.

### Configuration

- **Data Source:** The application uses the \"meteorites.csv\" dataset. Ensure the dataset is available in the root directory.
