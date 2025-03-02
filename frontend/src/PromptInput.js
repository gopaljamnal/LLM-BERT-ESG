import React, { useState } from 'react';
import axios from 'axios';

const PromptInput = () => {
  const [prompt, setPrompt] = useState('');
  const [response, setResponse] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      // Preprocess the prompt (if necessary)
      const preprocessedPrompt = preprocessPrompt(prompt);

      // Send the preprocessed prompt to the backend API
      const response = await axios.post('/generate_response', { prompt: preprocessedPrompt });

      // Format the response (if needed)
      const formattedResponse = formatResponse(response.data.response);

      setResponse(formattedResponse);
    } catch (error) {
      console.error(error);
    }
  };

  // Example preprocessing function
  const preprocessPrompt = (prompt) => {
    // Remove stop words, normalize text, etc.
    return prompt.toLowerCase().replace(/[^\w\s]/g, '');
  };

  // Example response formatting function
  const formatResponse = (response) => {
    // Add HTML formatting, or perform other modifications
    return `<p>${response}</p>`;
  };

  return (
    <div>
      <h2>Prompt-Based Model</h2>
      <form onSubmit={handleSubmit}>
        <label htmlFor="prompt">Enter your prompt:</label>
        <input type="text" id="prompt" value={prompt} onChange={(e) => setPrompt(e.target.value)} />
        <button type="submit">Submit</button>
      </form>
      <div dangerouslySetInnerHTML={{ __html: response }} />
    </div>
  );
};

export default PromptInput;