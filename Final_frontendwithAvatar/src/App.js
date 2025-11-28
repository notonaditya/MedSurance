// src/App.js
import React from 'react';
import AvatarSample from "./components/AvatarViewer";
import ChatInterface from './components/ChatInterface';
import './App.css';

function App() {
  return (
    <div className="app-container">
      <div className="avatar-container">
        <AvatarSample />
      </div>
      <div className="chat-container">
        <ChatInterface />
      </div>
    </div>
  );
}

export default App;
