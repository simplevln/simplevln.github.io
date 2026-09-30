const revealObserver = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add("visible");
        revealObserver.unobserve(entry.target);
      }
    });
  },
  { threshold: 0.12 }
);

document.querySelectorAll(".reveal").forEach((element) => revealObserver.observe(element));

const demoObserver = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry) => {
      const video = entry.target;
      if (entry.isIntersecting) {
        video.play().catch(() => {});
      } else {
        video.pause();
      }
    });
  },
  { threshold: 0.55 }
);

document.querySelectorAll(".demo-card video").forEach((video) => demoObserver.observe(video));

const copyButton = document.querySelector("#copy-citation");
const bibtex = document.querySelector("#bibtex");

copyButton.addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(bibtex.textContent);
    copyButton.textContent = "Copied ✓";
    setTimeout(() => {
      copyButton.textContent = "Copy BibTeX";
    }, 1800);
  } catch {
    const selection = window.getSelection();
    const range = document.createRange();
    range.selectNodeContents(bibtex);
    selection.removeAllRanges();
    selection.addRange(range);
    copyButton.textContent = "Selected";
  }
});
