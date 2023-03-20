import React, { useState } from "react";
import "./App.css";
import { FaFile } from "react-icons/fa";
import { FaPaperPlane } from 'react-icons/fa';

function App() {
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState("");
  const handleMessageSubmit = async (event) => {
    event.preventDefault();
    if (!inputValue.trim()) {
      return;
    }
    const messageObject = {
      text: inputValue.trim(),
      sender: "user",
      timestamp: new Date(),
    };
    setMessages((messages) => [...messages, messageObject]);
    setInputValue("");
    try {
      const response = await fetch("/api/gpt", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ message: messageObject.text }),
      });
      const data = await response.json();
      const botMessageObject = {
        text: data.answer,
        sender: "bot",
        timestamp: new Date(),
      };
      setMessages((messages) => [...messages, botMessageObject]);
    } catch (error) {
      console.error("Error:", error);
    }
  };
  return (
    <div className="app-container">
      <h1>LangChain & ChatGPT</h1>
      <div className="attachment-input">
        <input type="file" />
        <button>
          <FaFile />
        </button>
      </div>
      <br/>
      <div className="chat-container">
        {messages.map((message, index) => (
          <div
            key={index}
            className={`chat-message ${
              message.sender === "user" ? "user-message" : "bot-message"
            }`}
          >
            <div className="message-text">{message.text}</div>
            <div className="message-timestamp">
              {message.timestamp.toLocaleTimeString([], {
                hour: "2-digit",
                minute: "2-digit",
              })}
            </div>
          </div>
        ))}
      </div>
      <form onSubmit={handleMessageSubmit} className="input-form">
        <input
          type="text"
          placeholder="Enter your message..."
          value={inputValue}
          onChange={(event) => setInputValue(event.target.value)}
        />
        <button type="submit">
          <FaPaperPlane />
        </button>
      </form>
    </div>
  );
}

export default App;
