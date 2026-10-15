document.querySelectorAll('a[href^="#"]').forEach(link => {
  link.href = location.pathname + location.search + link.getAttribute('href');
});
// Keep each original heading and its description together during transitions.
document.querySelectorAll('.split-heading, .section-label, .pillars article, .score-grid article, .analysis-part, .statement-lead, .recipe').forEach(element => element.classList.add('enter'));
const motionPreference = matchMedia('(prefers-reduced-motion: reduce)');
const targets = [...document.querySelectorAll('.reveal, .enter')];
const observer = new IntersectionObserver(entries => {
  entries.forEach(entry => entry.target.classList.toggle('visible', entry.isIntersecting));
}, { threshold: 0, rootMargin: '0px' });
function configureMotion() {
  observer.disconnect();
  document.documentElement.classList.toggle('motion-ready', !motionPreference.matches);
  targets.forEach(element => {
    const rect = element.getBoundingClientRect();
    element.classList.toggle('visible', motionPreference.matches || (rect.bottom > 0 && rect.top < innerHeight));
    if (!motionPreference.matches) observer.observe(element);
  });
}
motionPreference.addEventListener('change', configureMotion);
configureMotion();
