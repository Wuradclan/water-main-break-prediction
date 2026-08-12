/* ==================================================================
   WATERBREAK AI — INTERACTIONS
   Vanilla JS only. No build step, no dependencies.
   Sections:
     1. Sticky navbar shadow
     2. Mobile hamburger menu
     3. Active nav link tracking on scroll
     4. Scroll-reveal animations
     5. Animated stat counters
     6. Hero HUD risk bar animation
     7. Interactive prediction demo (simulation)
     8. Back-to-top button
     9. Misc (footer year)
   ================================================================== */
(function () {
  'use strict';

  var prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  // The same script.js powers both index.html (en) and index-fr.html (fr).
  // <html lang="..."> tells us which copy to use for JS-generated text.
  var LANG = document.documentElement.lang === 'fr' ? 'fr' : 'en';
  var RISK_LABELS = {
    en: { high: 'HIGH RISK', medium: 'MEDIUM RISK', low: 'LOW RISK' },
    fr: { high: 'RISQUE ÉLEVÉ', medium: 'RISQUE MOYEN', low: 'RISQUE FAIBLE' }
  };

  /* ----------------------------------------------------------------
     1. STICKY NAVBAR — add shadow/background once the page scrolls
     ---------------------------------------------------------------- */
  var navbar = document.getElementById('navbar');

  function updateNavbarState() {
    if (window.scrollY > 8) {
      navbar.classList.add('is-scrolled');
    } else {
      navbar.classList.remove('is-scrolled');
    }
  }
  updateNavbarState();
  window.addEventListener('scroll', updateNavbarState, { passive: true });

  /* ----------------------------------------------------------------
     2. MOBILE HAMBURGER MENU
     ---------------------------------------------------------------- */
  var navToggle = document.getElementById('navToggle');
  var primaryNav = document.getElementById('primaryNav');

  function closeMobileMenu() {
    primaryNav.classList.remove('is-open');
    navToggle.setAttribute('aria-expanded', 'false');
  }

  navToggle.addEventListener('click', function () {
    var isOpen = primaryNav.classList.toggle('is-open');
    navToggle.setAttribute('aria-expanded', String(isOpen));
  });

  // Close the mobile menu whenever a nav link is used
  document.querySelectorAll('.nav-link').forEach(function (link) {
    link.addEventListener('click', closeMobileMenu);
  });

  /* ----------------------------------------------------------------
     3. ACTIVE NAV LINK ON SCROLL
     Highlights the nav item matching the section currently in view.
     ---------------------------------------------------------------- */
  var navLinks = Array.prototype.slice.call(document.querySelectorAll('.nav-link'));
  var trackedSections = navLinks
    .map(function (link) {
      var id = link.getAttribute('href');
      if (!id || id.charAt(0) !== '#') return null;
      var el = document.querySelector(id);
      return el ? { link: link, el: el } : null;
    })
    .filter(Boolean);

  if ('IntersectionObserver' in window && trackedSections.length) {
    var navObserver = new IntersectionObserver(
      function (entries) {
        entries.forEach(function (entry) {
          var match = trackedSections.filter(function (s) { return s.el === entry.target; })[0];
          if (!match) return;
          if (entry.isIntersecting) {
            navLinks.forEach(function (l) { l.classList.remove('is-active'); });
            match.link.classList.add('is-active');
          }
        });
      },
      { rootMargin: '-45% 0px -50% 0px', threshold: 0 }
    );
    trackedSections.forEach(function (s) { navObserver.observe(s.el); });
  }

  /* ----------------------------------------------------------------
     4. SCROLL-REVEAL ANIMATIONS
     Elements with class "reveal" fade + slide up once visible.
     ---------------------------------------------------------------- */
  var revealEls = document.querySelectorAll('.reveal');

  if (prefersReducedMotion || !('IntersectionObserver' in window)) {
    revealEls.forEach(function (el) { el.classList.add('in-view'); });
  } else {
    var revealObserver = new IntersectionObserver(
      function (entries, observer) {
        entries.forEach(function (entry) {
          if (entry.isIntersecting) {
            entry.target.classList.add('in-view');
            observer.unobserve(entry.target);
          }
        });
      },
      { threshold: 0.12, rootMargin: '0px 0px -60px 0px' }
    );
    revealEls.forEach(function (el) { revealObserver.observe(el); });
  }

  /* ----------------------------------------------------------------
     5. ANIMATED STAT COUNTERS
     Counts up from 0 to data-target once the stat scrolls into view.
     ---------------------------------------------------------------- */
  function animateCounter(el) {
    var target = parseInt(el.getAttribute('data-target'), 10) || 0;
    if (prefersReducedMotion) {
      el.textContent = String(target);
      return;
    }
    var duration = 1200;
    var start = null;

    function step(timestamp) {
      if (start === null) start = timestamp;
      var progress = Math.min((timestamp - start) / duration, 1);
      var eased = 1 - Math.pow(1 - progress, 3); // ease-out-cubic
      el.textContent = String(Math.round(eased * target));
      if (progress < 1) {
        window.requestAnimationFrame(step);
      } else {
        el.textContent = String(target);
      }
    }
    window.requestAnimationFrame(step);
  }

  var counterEls = document.querySelectorAll('[data-counter]');
  if ('IntersectionObserver' in window && counterEls.length) {
    var counterObserver = new IntersectionObserver(
      function (entries, observer) {
        entries.forEach(function (entry) {
          if (entry.isIntersecting) {
            animateCounter(entry.target);
            observer.unobserve(entry.target);
          }
        });
      },
      { threshold: 0.4 }
    );
    counterEls.forEach(function (el) { counterObserver.observe(el); });
  } else {
    counterEls.forEach(animateCounter);
  }

  /* ----------------------------------------------------------------
     6. HERO HUD RISK BAR
     Animates the illustrative "Pipeline Risk Analysis" gauge once
     the hero card is in view.
     ---------------------------------------------------------------- */
  var heroRiskValue = document.querySelector('[data-animate-risk]');
  var heroRiskFill = document.querySelector('[data-risk-fill]');

  function runHeroRiskAnimation() {
    if (!heroRiskValue || !heroRiskFill) return;
    var target = parseInt(heroRiskValue.getAttribute('data-animate-risk'), 10) || 0;
    heroRiskFill.style.width = target + '%';

    if (prefersReducedMotion) {
      heroRiskValue.textContent = target + '%';
      return;
    }
    var duration = 1200;
    var start = null;
    function step(timestamp) {
      if (start === null) start = timestamp;
      var progress = Math.min((timestamp - start) / duration, 1);
      heroRiskValue.textContent = Math.round(progress * target) + '%';
      if (progress < 1) window.requestAnimationFrame(step);
    }
    window.requestAnimationFrame(step);
  }

  // Small delay so the animation feels intentional once the hero paints
  window.setTimeout(runHeroRiskAnimation, 350);

  /* ----------------------------------------------------------------
     7. INTERACTIVE PREDICTION DEMO (client-side simulation)
     This is NOT the real ML model — it is a transparent, explainable
     formula meant purely to demonstrate the UX of the real /predict
     endpoint. The real prediction is produced by the FastAPI service.
     ---------------------------------------------------------------- */
  var demoForm = document.getElementById('demoForm');
  var demoResultValue = document.getElementById('demoResultValue');
  var demoResultBar = document.getElementById('demoResultBar');
  var demoResultStatus = document.getElementById('demoResultStatus');
  var demoResultCard = document.getElementById('demoResultCard');

  // Illustrative per-material weight — cast iron is the oldest, most
  // brittle material typically found in KW's historical inventory.
  var MATERIAL_COEFFICIENTS = {
    'cast-iron': 24,
    'ductile-iron': 10,
    'steel': 14,
    'pvc': 3
  };

  function computeSimulatedRisk(material, diameter, age, breaks) {
    var ageFactor = age * 0.55;
    var breaksFactor = breaks * 11;
    var materialFactor = MATERIAL_COEFFICIENTS[material] !== undefined ? MATERIAL_COEFFICIENTS[material] : 8;
    // Smaller-diameter mains fail slightly more often in the KW dataset.
    var diameterFactor = diameter < 150 ? 6 : diameter > 350 ? -4 : 0;

    var raw = ageFactor + breaksFactor + materialFactor + diameterFactor;
    // Clamp to a believable 2%–97% probability band.
    return Math.max(2, Math.min(97, Math.round(raw)));
  }

  function riskLevel(risk) {
    if (risk >= 67) return 'high';
    if (risk >= 34) return 'medium';
    return 'low';
  }

  function animateDemoBar(target) {
    demoResultBar.style.width = target + '%';
    if (prefersReducedMotion) {
      demoResultValue.textContent = target + '%';
      return;
    }
    var duration = 900;
    var start = null;
    function step(timestamp) {
      if (start === null) start = timestamp;
      var progress = Math.min((timestamp - start) / duration, 1);
      demoResultValue.textContent = Math.round(progress * target) + '%';
      if (progress < 1) window.requestAnimationFrame(step);
    }
    window.requestAnimationFrame(step);
  }

  if (demoForm) {
    demoForm.addEventListener('submit', function (event) {
      event.preventDefault();

      var material = document.getElementById('demoMaterial').value;
      var diameter = parseFloat(document.getElementById('demoDiameter').value) || 0;
      var age = parseFloat(document.getElementById('demoAge').value) || 0;
      var breaks = parseFloat(document.getElementById('demoBreaks').value) || 0;

      var risk = computeSimulatedRisk(material, diameter, age, breaks);
      var level = riskLevel(risk);

      animateDemoBar(risk);

      demoResultStatus.textContent = RISK_LABELS[LANG][level];
      demoResultStatus.classList.remove('is-low', 'is-medium', 'is-high');
      demoResultStatus.classList.add('is-' + level);

      demoResultCard.classList.remove('is-pulsing');
      void demoResultCard.offsetWidth; // restart the pulse animation
      demoResultCard.classList.add('is-pulsing');
    });
  }

  /* ----------------------------------------------------------------
     8. BACK TO TOP BUTTON
     ---------------------------------------------------------------- */
  var backToTop = document.getElementById('backToTop');

  function updateBackToTop() {
    if (window.scrollY > 640) {
      backToTop.classList.add('is-visible');
    } else {
      backToTop.classList.remove('is-visible');
    }
  }
  updateBackToTop();
  window.addEventListener('scroll', updateBackToTop, { passive: true });

  backToTop.addEventListener('click', function () {
    window.scrollTo({ top: 0, behavior: prefersReducedMotion ? 'auto' : 'smooth' });
  });

  /* ----------------------------------------------------------------
     9. MISC — footer year
     ---------------------------------------------------------------- */
  var yearEl = document.getElementById('year');
  if (yearEl) yearEl.textContent = String(new Date().getFullYear());

  /* ----------------------------------------------------------------
     10. ACCESSIBILITY — hide purely decorative icons from
     assistive technology (every icon here sits next to a visible
     text label, so the icon itself carries no extra information).
     ---------------------------------------------------------------- */
  document.querySelectorAll('svg.icon').forEach(function (svg) {
    svg.setAttribute('aria-hidden', 'true');
    svg.setAttribute('focusable', 'false');
  });

})();
