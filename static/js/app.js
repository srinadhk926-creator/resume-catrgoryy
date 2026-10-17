/**
 * ResumeAI - Frontend Drag-and-Drop & Client Validation
 */

document.addEventListener("DOMContentLoaded", () => {
  const dropZone = document.getElementById("drop-zone");
  const fileInput = document.getElementById("resume-file");
  const filePreview = document.getElementById("file-preview");
  const previewName = document.getElementById("preview-name");
  const previewRemove = document.getElementById("preview-remove");
  const submitBtn = document.getElementById("submit-btn");
  const uploadForm = document.getElementById("upload-form");

  if (!dropZone || !fileInput) return;

  const MAX_FILE_SIZE = 5 * 1024 * 1024; // 5MB
  const ALLOWED_EXTS = ["pdf", "docx"];

  function handleFileSelected(file) {
    if (!file) return;

    const ext = file.name.split(".").pop().toLowerCase();
    if (!ALLOWED_EXTS.includes(ext)) {
      alert(`Unsupported file format (.${ext}). Please select a PDF or DOCX file.`);
      fileInput.value = "";
      filePreview.style.display = "none";
      return;
    }

    if (file.size > MAX_FILE_SIZE) {
      const mbSize = (file.size / (1024 * 1024)).toFixed(2);
      alert(`File size (${mbSize} MB) exceeds the 5 MB limit. Please choose a smaller file.`);
      fileInput.value = "";
      filePreview.style.display = "none";
      return;
    }

    // Display preview
    previewName.textContent = `${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
    filePreview.style.display = "flex";
  }

  // File input change
  fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFileSelected(e.target.files[0]);
    }
  });

  // Drag-and-drop events
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
      fileInput.files = e.dataTransfer.files;
      handleFileSelected(e.dataTransfer.files[0]);
    }
  });

  // Remove selected file
  if (previewRemove) {
    previewRemove.addEventListener("click", (e) => {
      e.stopPropagation();
      fileInput.value = "";
      filePreview.style.display = "none";
    });
  }

  // Loading state on form submit
  if (uploadForm && submitBtn) {
    uploadForm.addEventListener("submit", () => {
      submitBtn.disabled = true;
      submitBtn.innerHTML = `<span>Analyzing Resume with ML... ⏳</span>`;
    });
  }
});
