/**
 * ResumeAI - Frontend Drag-and-Drop, Tab Switching & Client Validation
 */

document.addEventListener("DOMContentLoaded", () => {
  const dropZone = document.getElementById("drop-zone");
  const fileInput = document.getElementById("resume-file");
  const filePreview = document.getElementById("file-preview");
  const previewName = document.getElementById("preview-name");
  const previewRemove = document.getElementById("preview-remove");
  const submitBtn = document.getElementById("submit-btn");
  const uploadForm = document.getElementById("upload-form");
  const resumeText = document.getElementById("resume-text");

  // Tab switching elements
  const tabFile = document.getElementById("tab-file");
  const tabText = document.getElementById("tab-text");
  const panelFile = document.getElementById("panel-file");
  const panelText = document.getElementById("panel-text");

  if (tabFile && tabText && panelFile && panelText) {
    tabFile.addEventListener("click", () => {
      tabFile.classList.add("active");
      tabFile.setAttribute("aria-selected", "true");
      tabText.classList.remove("active");
      tabText.setAttribute("aria-selected", "false");
      panelFile.style.display = "block";
      panelText.style.display = "none";
      if (resumeText) resumeText.value = "";
    });

    tabText.addEventListener("click", () => {
      tabText.classList.add("active");
      tabText.setAttribute("aria-selected", "true");
      tabFile.classList.remove("active");
      tabFile.setAttribute("aria-selected", "false");
      panelText.style.display = "block";
      panelFile.style.display = "none";
      if (fileInput) fileInput.value = "";
      if (filePreview) filePreview.style.display = "none";
    });
  }

  // Mobile navigation menu toggle
  const mobileNavToggle = document.getElementById("mobile-nav-toggle");
  const navLinks = document.getElementById("nav-links");
  if (mobileNavToggle && navLinks) {
    mobileNavToggle.addEventListener("click", () => {
      const isOpen = navLinks.classList.toggle("open");
      mobileNavToggle.setAttribute("aria-expanded", isOpen ? "true" : "false");
    });
  }

  // Password visibility toggle helper
  function setupPasswordToggle(btnId, inputId) {
    const btn = document.getElementById(btnId);
    const input = document.getElementById(inputId);
    if (!btn || !input) return;

    btn.addEventListener("click", (e) => {
      e.preventDefault();
      const isPassword = input.type === "password";
      input.type = isPassword ? "text" : "password";
      const eyeOpen = btn.querySelector(".eye-open");
      const eyeClosed = btn.querySelector(".eye-closed");
      if (eyeOpen && eyeClosed) {
        eyeOpen.style.display = isPassword ? "none" : "block";
        eyeClosed.style.display = isPassword ? "block" : "none";
      }
      btn.setAttribute("aria-label", isPassword ? "Hide password" : "Show password");
    });
  }

  setupPasswordToggle("toggle-password", "password");
  setupPasswordToggle("toggle-password-reg", "password");

  const MAX_FILE_SIZE = 5 * 1024 * 1024; // 5 MB
  const ALLOWED_EXTS = ["pdf", "docx"];

  function handleFileSelected(file) {
    if (!file) return;

    const ext = file.name.split(".").pop().toLowerCase();
    if (!ALLOWED_EXTS.includes(ext)) {
      alert(`Unsupported file format (.${ext}). Please select a PDF or DOCX file.`);
      if (fileInput) fileInput.value = "";
      if (filePreview) filePreview.style.display = "none";
      return;
    }

    if (file.size > MAX_FILE_SIZE) {
      const mbSize = (file.size / (1024 * 1024)).toFixed(2);
      alert(`File size (${mbSize} MB) exceeds the 5 MB limit. Please select a smaller document.`);
      if (fileInput) fileInput.value = "";
      if (filePreview) filePreview.style.display = "none";
      return;
    }

    // Display selected file preview
    if (previewName && filePreview) {
      previewName.textContent = `${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
      filePreview.style.display = "flex";
    }
  }

  // File input change event
  if (fileInput) {
    fileInput.addEventListener("change", (e) => {
      if (e.target.files && e.target.files[0]) {
        handleFileSelected(e.target.files[0]);
      }
    });
  }

  // Drag-and-drop events
  if (dropZone) {
    ["dragenter", "dragover"].forEach((eventName) => {
      dropZone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropZone.classList.add("dragover");
      });
    });

    ["dragleave", "drop"].forEach((eventName) => {
      dropZone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropZone.classList.remove("dragover");
      });
    });

    dropZone.addEventListener("drop", (e) => {
      if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        if (fileInput) fileInput.files = e.dataTransfer.files;
        handleFileSelected(e.dataTransfer.files[0]);
      }
    });
  }

  // Remove selected file
  if (previewRemove) {
    previewRemove.addEventListener("click", (e) => {
      e.stopPropagation();
      if (fileInput) fileInput.value = "";
      if (filePreview) filePreview.style.display = "none";
    });
  }

  // Form submit loading state
  const loadingOverlay = document.getElementById("loading-overlay");

  function resetUploadState() {
    if (submitBtn) {
      submitBtn.disabled = false;
      submitBtn.innerHTML = `
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon></svg>
        <span>Analyze &amp; Classify Resume</span>
      `;
    }
    if (loadingOverlay) {
      loadingOverlay.style.display = "none";
      loadingOverlay.setAttribute("aria-busy", "false");
    }
  }

  // Handle browser back/forward restoration (bfcache)
  window.addEventListener("pageshow", (event) => {
    resetUploadState();
    const loginBtn = document.getElementById("login-submit-btn");
    if (loginBtn) {
      loginBtn.disabled = false;
      loginBtn.textContent = "Sign In to Workspace";
    }
    const regBtn = document.getElementById("register-submit-btn");
    if (regBtn) {
      regBtn.disabled = false;
      regBtn.textContent = "Create Free Account";
    }
  });

  if (uploadForm && submitBtn) {
    uploadForm.addEventListener("submit", (e) => {
      const hasFile = fileInput && fileInput.files && fileInput.files.length > 0;
      const hasText = resumeText && resumeText.value.trim().length >= 20;

      if (!hasFile && !hasText) {
        e.preventDefault();
        alert("Please select a valid resume document (.pdf or .docx) or paste at least 20 characters of resume text.");
        return;
      }

      // Legitimate client-side submission: Activate loading overlay
      submitBtn.disabled = true;
      submitBtn.innerHTML = `<span>Processing Document...</span>`;

      if (loadingOverlay) {
        loadingOverlay.style.display = "flex";
        loadingOverlay.setAttribute("aria-busy", "true");
      }
    });
  }

  // Auth forms submit loading states
  const loginForm = document.getElementById("login-form");
  const loginSubmitBtn = document.getElementById("login-submit-btn");
  if (loginForm && loginSubmitBtn) {
    loginForm.addEventListener("submit", () => {
      loginSubmitBtn.disabled = true;
      loginSubmitBtn.textContent = "Signing In...";
    });
  }

  const registerForm = document.getElementById("register-form");
  const registerSubmitBtn = document.getElementById("register-submit-btn");
  if (registerForm && registerSubmitBtn) {
    registerForm.addEventListener("submit", () => {
      registerSubmitBtn.disabled = true;
      registerSubmitBtn.textContent = "Creating Account...";
    });
  }
});
