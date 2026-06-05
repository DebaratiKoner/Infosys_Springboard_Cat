/* =========================================================
   PAGE NAVIGATION (SPA LOGIC)
   ========================================================= */

function showPage(pageId) {
  const pages = document.querySelectorAll(".page");

  pages.forEach(page => page.classList.add("hidden"));

  const target = document.getElementById(pageId);
  if (!target) {
    console.error("Page not found:", pageId);
    return;
  }

  target.classList.remove("hidden");
}


/* =========================================================
   CHATBOT TOGGLE
   ========================================================= */

function toggleChat() {
  const chat = document.getElementById("chatWindow");
  const box = document.getElementById("chatBox");

  if (chat.style.display === "block") {
    chat.style.display = "none";
  } else {
    chat.style.display = "block";

    if (box.innerHTML.trim() === "") {
      box.innerHTML = `<div><b>MeowBot:</b> Hi! Ask me about cat emotions, food, play, or behavior.</div>`;
    }
  }
}


/* =========================================================
   CHATBOT LOGIC (MeowBot)
   ========================================================= */

function sendChat() {
  const input = document.getElementById("chatInput");
  const box = document.getElementById("chatBox");

  const userText = input.value.trim();
  if (userText === "") return;

  const text = userText.toLowerCase();
  box.innerHTML += `<div><b>You:</b> ${userText}</div>`;

  let reply = "Cats communicate mainly through body language and sound.";

  if (text.includes("angry") || text.includes("mad")) {
    reply = "Angry cats need space. Avoid touching and remove stress triggers.";
  } else if (text.includes("scared") || text.includes("fear") || text.includes("afraid")) {
    reply = "Scared cats need safe hiding spots and calm surroundings.";
  } else if (text.includes("happy")) {
    reply = "A happy cat enjoys play, gentle affection, and treats!";
  } else if (text.includes("sad") || text.includes("depressed")) {
    reply = "A sad cat may be stressed or unwell. Monitor behavior closely.";
  } else if (text.includes("food") || text.includes("eat")) {
    reply = "Balanced nutrition supports both emotional and physical health.";
  } else if (text.includes("play") || text.includes("toy")) {
    reply = "Interactive toys help release energy and reduce stress.";
  } else if (text.includes("sleep")) {
    reply = "Cats sleep a lot! It helps them conserve energy and feel secure.";
  }

  box.innerHTML += `<div><b>MeowBot:</b> ${reply}</div>`;
  input.value = "";
  box.scrollTop = box.scrollHeight;
}


/* =========================================================
   CONTACT FORM MESSAGE
   ========================================================= */

function sendMessage() {
  alert("Your message has been sent successfully!");

  document.querySelectorAll("#contact input, #contact textarea")
    .forEach(el => el.value = "");
}


/* =========================================================
   ANALYSIS PAGE
   ========================================================= */

async function predictImage() {
  const fileInput = document.getElementById("imageInput");
  const result = document.getElementById("imageResult");
  const button = document.getElementById("imagePredictBtn");

  if (fileInput.files.length === 0) {
    result.innerText = "Please upload an image.";
    return;
  }

  const formData = new FormData();
  formData.append("file", fileInput.files[0]);

  result.innerText = "Analyzing image...";
  button.disabled = true;

  try {
    const response = await fetch("/predict_image", {
      method: "POST",
      body: formData
    });
    const data = await response.json();

    if (data.error) {
      result.innerText = data.error;
    } else {
      result.innerHTML = `<b>Emotion:</b> ${data.emotion}<br><b>Confidence:</b> ${data.confidence}%`;
    }
  } catch (error) {
    result.innerText = "Could not analyze image. Please try again.";
  } finally {
    button.disabled = false;
  }
}

async function predictAudio() {
  const fileInput = document.getElementById("audioInput");
  const result = document.getElementById("audioResult");
  const button = document.getElementById("audioPredictBtn");

  if (fileInput.files.length === 0) {
    result.innerText = "Please upload an audio file.";
    return;
  }

  const formData = new FormData();
  formData.append("file", fileInput.files[0]);

  result.innerText = "Analyzing audio...";
  button.disabled = true;

  try {
    const response = await fetch("/predict_audio", {
      method: "POST",
      body: formData
    });
    const data = await response.json();

    if (data.error) {
      result.innerText = data.error;
    } else {
      result.innerHTML = `<b>Emotion:</b> ${data.emotion}<br><b>Confidence:</b> ${data.confidence}%`;
    }
  } catch (error) {
    result.innerText = "Could not analyze audio. Please try again.";
  } finally {
    button.disabled = false;
  }
}

function previewImage() {
  const fileInput = document.getElementById("imageInput");
  const preview = document.getElementById("imagePreview");

  if (fileInput.files.length === 0) return;

  preview.src = URL.createObjectURL(fileInput.files[0]);
  preview.style.display = "block";
}

function previewAudio() {
  const fileInput = document.getElementById("audioInput");
  const audio = document.getElementById("audioPreview");

  if (!fileInput.files || fileInput.files.length === 0) return;

  audio.src = URL.createObjectURL(fileInput.files[0]);
  audio.style.display = "block";
}
