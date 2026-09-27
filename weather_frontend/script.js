// WeatherGPT frontend logic — talks to the FastAPI backend at API_URL,
// handles language switching and voice input (Web Speech API).
 
const API_URL = "http://localhost:8000/chat";
const HEALTH_URL = "http://localhost:8000/health";
 
let lang = "en";
let defaultLocation = ""; // fallback city if none is mentioned in the message
 
const messagesEl = document.getElementById('messages');
const input = document.getElementById('msgInput');
const micBtn = document.getElementById('micBtn');
const form = document.getElementById('composerForm');
const statusDot = document.getElementById('statusDot');
const statusText = document.getElementById('statusText');
const apiUrlLabel = document.getElementById('apiUrlLabel');
 
apiUrlLabel.textContent = API_URL;
 
const GREETING = {
  en: "Hi! Ask me about weather, forecasts, or alerts for any place — e.g. \"Will it rain in Lucknow tomorrow?\"",
  hi: "Namaste! Mujhse kisi bhi jagah ka mausam, forecast ya alert poochho — jaise \"kal Lucknow mein baarish hogi kya?\""
};
 
const PLACEHOLDER = {
  en: "e.g. Will it rain in Lucknow tomorrow?",
  hi: "jaise: kal Lucknow mein baarish hogi kya?"
};
 
const OFFLINE_MSG = {
  en: "Couldn't reach the backend. Is the server running? (uvicorn main:app --reload)",
  hi: "Backend se connect nahi ho paaya. Server chalu hai kya? (uvicorn main:app --reload)"
};
 
function addMessage(text, sender, isAlert = false){
  const div = document.createElement('div');
  div.className = 'msg ' + sender + (isAlert ? ' alert' : '');
  div.textContent = text;
  messagesEl.appendChild(div);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}
 
addMessage(GREETING[lang], 'bot');
 
// ---- Language toggle ----
document.querySelectorAll('.lang-toggle button').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.lang-toggle button').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    lang = btn.dataset.lang;
    input.placeholder = PLACEHOLDER[lang];
  });
});
 
// ---- Backend health check ----
async function checkBackend(){
  try{
    const res = await fetch(HEALTH_URL, { method: 'GET' });
    if(res.ok){
      statusDot.classList.add('ok');
      statusDot.classList.remove('down');
      statusText.textContent = 'Backend connected';
    } else {
      throw new Error('bad status');
    }
  }catch(err){
    statusDot.classList.add('down');
    statusDot.classList.remove('ok');
    statusText.textContent = 'Backend offline';
  }
}
checkBackend();
setInterval(checkBackend, 15000);
 
// ---- Sending a message ----
async function sendMessage(){
  const text = input.value.trim();
  if(!text) return;
  addMessage(text, 'user');
  input.value = '';
 
  try{
    const res = await fetch(API_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text, lang: lang, default_location: defaultLocation })
    });
    const data = await res.json();
    addMessage(data.reply, 'bot', data.alerts && data.alerts.length > 0);
  }catch(err){
    addMessage(OFFLINE_MSG[lang], 'bot');
  }
}
 
form.addEventListener('submit', (e) => {
  e.preventDefault();
  sendMessage();
});
 
// ---- Voice input (Web Speech API — Chrome/Edge) ----
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
if(SpeechRecognition){
  const recognition = new SpeechRecognition();
  recognition.continuous = false;
  recognition.interimResults = false;
 
  micBtn.addEventListener('click', () => {
    recognition.lang = lang === 'hi' ? 'hi-IN' : 'en-IN';
    recognition.start();
    micBtn.classList.add('listening');
  });
  recognition.onresult = (event) => {
    input.value = event.results[0][0].transcript;
    micBtn.classList.remove('listening');
    sendMessage();
  };
  recognition.onerror = () => micBtn.classList.remove('listening');
  recognition.onend = () => micBtn.classList.remove('listening');
} else {
  micBtn.style.display = 'none';
}