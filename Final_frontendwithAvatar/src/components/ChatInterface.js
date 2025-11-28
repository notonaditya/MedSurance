// src/components/ChatInterface.js
import React, { useState, useEffect, useRef } from 'react';
import './ChatInterface.css';
export default function ChatInterface() {
  const [chatHistory, setChatHistory] = useState([]);
  const [medicalRecord, setMedicalRecord] = useState({});
  const [currentField, setCurrentField] = useState(null);
  const [userInput, setUserInput] = useState('');
  const [waitingForResponse, setWaitingForResponse] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const recognitionRef = useRef(null);
  const lastTranscriptRef = useRef('');
  // This ref will prevent the mic from restarting while bot is speaking.
  const suppressListeningRef = useRef(false);


  // Play audio from a base64 string and return the audio element
  function playAudio(audioData) {
    if (!audioData) return null;
    try {
      const audio = new Audio(`data:audio/mp3;base64,${audioData}`);
      audio.play().catch(e => console.error('Error playing audio:', e));
      return audio;
    } catch (e) {
      console.error('Error creating audio:', e);
      return null;
    }
  }

  // Setup Speech Recognition on mount
  useEffect(() => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (SpeechRecognition) {
      const recognition = new SpeechRecognition();
      recognition.lang = 'en-US';
      recognition.continuous = false;
      recognition.interimResults = true;

      recognition.onresult = (event) => {
        const transcript = Array.from(event.results)
          .map(result => result[0].transcript)
          .join('');
        setUserInput(transcript);
        lastTranscriptRef.current = transcript;
      };

      recognition.onend = () => {
        setIsListening(false);
        // Do not restart if bot's audio is playing
        if (suppressListeningRef.current) return;
        // If a transcript exists and we’re not waiting, submit it.
        if (lastTranscriptRef.current.trim() && !waitingForResponse) {
          submitAnswer(lastTranscriptRef.current);
          lastTranscriptRef.current = '';
        } else if (!waitingForResponse) {
          startListening();
        }
      };

      recognition.onerror = (event) => {
        console.error("Speech recognition error:", event.error);
        setIsListening(false);
      };

      recognitionRef.current = recognition;
    }
  }, [waitingForResponse]);

  // Start conversation on mount
  useEffect(() => {
  let isMounted = true; // Prevent unnecessary calls

  fetch('/api/start', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' }
  })
    .then((res) => res.json())
    .then((data) => {
      if (isMounted) {
        setMedicalRecord(data.medical_record);
        getNextQuestion(data.medical_record);
      }
    })
    .catch((error) => console.error('Error starting conversation:', error));

  return () => {
    isMounted = false; // Cleanup to prevent
    // extra calls
  };
}, []);



  // Start listening (if not already)
  const startListening = () => {
    if (recognitionRef.current && !isListening && !waitingForResponse) {
      try {
        recognitionRef.current.start();
        setIsListening(true);
      } catch (error) {
        console.error('Error starting recognition:', error);
      }
    }
  };

  // Helper to re-enable listening after audio finishes (plus an extra delay)
  const scheduleListening = (audioEl) => {
    if (audioEl) {
      audioEl.onended = () => {
        setTimeout(() => {
          suppressListeningRef.current = false;
          startListening();
        }, 1500);
      };
    } else {
      // Fallback if there's no audio element
      setTimeout(() => {
        suppressListeningRef.current = false;
        startListening();
      }, 1500);
    }
  };

  // Fetch next question from the backend and update assistant message
  function getNextQuestion(record) {
    fetch('/api/next_question', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ medical_record: record })
    })
      .then((res) => res.json())
      .then((data) => {
        if (data.current_field && data.question) {
          setCurrentField(data.current_field);
          setChatHistory(prev => {
            if (prev.length && prev[prev.length - 1].sender === 'assistant') {
              return [...prev.slice(0, -1), { sender: 'assistant', message: data.question }];
            }
            return [...prev, { sender: 'assistant', message: data.question }];
          });
          // Suppress listening while bot speaks
          suppressListeningRef.current = true;
          const audioEl = playAudio(data.audio_data);
          setUserInput('');
          setWaitingForResponse(false);
          scheduleListening(audioEl);
        }
      })
      .catch((error) => {
        console.error('Error fetching next question:', error);
        setWaitingForResponse(false);
        suppressListeningRef.current = true;
        setTimeout(() => {
          suppressListeningRef.current = false;
          startListening();
        }, 1500);
      });
  }

  // Submit the user's answer to the backend
  function submitAnswer(answer) {
    if (!answer || !answer.trim() || waitingForResponse) return;
    const cleanAnswer = answer.trim();
    setWaitingForResponse(true);
    setChatHistory(prev => [...prev, { sender: 'user', message: cleanAnswer }]);

    // Stop listening immediately
    if (recognitionRef.current && isListening) {
      recognitionRef.current.stop();
      setIsListening(false);
    }

    fetch('/api/submit_answer', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        current_field: currentField,
        answer: cleanAnswer,
        medical_record: medicalRecord
      })
    })
      .then((res) => res.json())
      .then((data) => {
        let audioEl = null;
        if (data.status === 'success') {
          setMedicalRecord(data.medical_record);
          setCurrentField(data.current_field);
          setChatHistory(prev => {
            if (prev.length && prev[prev.length - 1].sender === 'assistant') {
              return [...prev.slice(0, -1), { sender: 'assistant', message: data.question }];
            }
            return [...prev, { sender: 'assistant', message: data.question }];
          });
          audioEl = playAudio(data.audio_data);
          setUserInput('');
        } else if (data.status === 'complete') {
          setMedicalRecord(data.medical_record);
          // setChatHistory(prev => [...prev, { sender: 'assistant', message: data.summary }]);
          audioEl = playAudio(data.audio_data);
        } else if (data.status === 'error') {
          setChatHistory(prev => [...prev, { sender: 'assistant', message: data.message }]);
          audioEl = playAudio(data.audio_data);
        }
        suppressListeningRef.current = true;
        scheduleListening(audioEl);
      })
      .catch((error) => {
        console.error('Error submitting answer:', error);
        suppressListeningRef.current = true;
        setTimeout(() => {
          suppressListeningRef.current = false;
          startListening();
        }, 1500);
      })
      .finally(() => {
        setWaitingForResponse(false);
      });
  }

  return (
    <div className="chat-interface">
      <h2 className="headerss"><center>Medical Interview</center></h2>
      <div className="chat-history">
        {chatHistory.map((msg, idx) => (
          <div key={idx} className={`message ${msg.sender}`}>
            <strong>{msg.sender === 'assistant' ? 'Assistant' : 'You'}:</strong> {msg.message}
          </div>
        ))}
      </div>
      <div className="chat-input">
        <div className="status-indicator">
          {isListening ? 'Listening...' : waitingForResponse ? 'Processing...' : 'Ready'}
        </div>
        <input
          type="text"
          value={userInput}
          onChange={(e) => setUserInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !waitingForResponse) {
              if (recognitionRef.current && isListening) {
                recognitionRef.current.stop();
                setIsListening(false);
              }
              submitAnswer(userInput);
              setUserInput('');
              lastTranscriptRef.current = '';
            }
          }}
          onBlur={() => {
            if (!waitingForResponse) {
              submitAnswer(userInput);
              setUserInput('');
              lastTranscriptRef.current = '';
            }
          }}
          autoFocus
          placeholder={isListening ? 'Listening...' : 'Type your answer...'}
        />
      </div>
    </div>
  );
}
