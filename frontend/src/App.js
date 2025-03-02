import React, { useState } from 'react';
import axios from 'axios';
import Plot from 'react-plotly.js';

function App() {
  const [url, setUrl] = useState('');
  const [chartData, setChartData] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const response = await axios.post('http://127.0.0.1:5000/classify', { url });
      const result = response.data.score;
      console.log(result)

      // Process the result for plotting
      const labels = Object.keys(result);
      const scores = labels.map(label => result[label]);

      setChartData({
        labels: labels,
        scores: scores,
      });

    } catch (error) {
      console.error("Error fetching data:", error.response || error.message || error);
    }
  };

  return (
      <div className="App">
        <h1>ESG Report Classification</h1>
        <form onSubmit={handleSubmit}>
          <label>
              Emter ESG Report URL, it may take a few minutes as BERT analyzes the document.)

            <input
                type="text"
                value={url}
                onChange={(e) => setUrl(e.target.value)}
                placeholder="https://www.responsibilityreports.com/Click/2534"
            />
          </label>
          <button type="submit">Classify</button>
        </form>

        {chartData && (
            <Plot
                data={[
                  {
                    x: chartData.labels,
                    y: chartData.scores,
                    type: 'bar',
                    marker: {color: 'blue'},
                  },
                ]}
                layout={{
                  title: 'ESG Report Classification Scores',
                  xaxis: {title: 'Labels'},
                  yaxis: {title: 'Score'},
                }}
            />
        )}
        <footer className="footer">
          <p>&copy; 2025 ESG Classification Tool. All rights reserved.</p>
        </footer>
      </div>
  );
}

export default App;

