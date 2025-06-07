const johnSelectorBtn = document.querySelector('#john-selector');
const janeSelectorBtn = document.querySelector('#jane-selector');
const chatHeader = document.querySelector('.chat-header');
const chatMessages = document.querySelector('.chat-message');
const chatInputForm = document.querySelector('.chat-input-form');
const chatInput = document.querySelector('.chat-input');
const clearChatBtn = document.querySelector('.clear-chat-button');

const messages = JSON.parse(localStorage.getItem('messages')) || [];
const createChatMessageElement = (message) => `
<div class="message ${message.sender === 'john' ? 'blue-bg' : 'gray-bg'}">
    <div class="message-sender">${message.sender}</div>
    <div class="message-text">${message.text}</div>
    <div class="message-timestamp">${message.timestamp}</div>
</div>
`;

window.onload = () => {
    messages.forEach(message => {
        chatMessages.innerHTML += createChatMessageElement(message);
    });
};

let messageSender = 'john';
const updateMessageSender = (name) => {
    messageSender = name;
    chatHeader.innerText = `${messageSender} Chatting...`;
    chatInput.placeholder = `Type here, ${messageSender}`;
    if (name === 'john') {
        johnSelectorBtn.classList.add('active-person');
        janeSelectorBtn.classList.remove('active-person');
    } else if (name === 'jane') {
        janeSelectorBtn.classList.add('active-person');
        johnSelectorBtn.classList.remove('active-person');
    }
    chatInput.focus();
};

johnSelectorBtn.onclick = () => updateMessageSender('john');
janeSelectorBtn.onclick = () => updateMessageSender('jane');

const sendMessage = (e) => {
    e.preventDefault();
    const timestamp = new Date().toLocaleString('en-US', { hour: 'numeric', minute: 'numeric', hour12: true });
    const message = {
        sender: messageSender,
        text: chatInput.value,
        timestamp,
    };
    messages.push(message);
    localStorage.setItem('messages', JSON.stringify(messages));
    chatMessages.innerHTML += createChatMessageElement(message);

    chatInputForm.reset();
    chatMessages.scrollTop = chatMessages.scrollHeight;
};

chatInputForm.addEventListener('submit', sendMessage);

clearChatBtn.addEventListener('click', () => {
    localStorage.clear();
    chatMessages.innerHTML = '';
});
