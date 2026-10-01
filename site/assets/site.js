// Blackbody site: menus, the before/after slider, video playback, copy buttons, the guide contents.
(function () {
  'use strict';

  // header menu on phones
  var toggle = document.querySelector('.nav-toggle');
  var nav = document.querySelector('.site-nav');
  if (toggle && nav) {
    toggle.addEventListener('click', function () {
      var open = nav.classList.toggle('open');
      toggle.setAttribute('aria-expanded', open ? 'true' : 'false');
    });
  }

  // guide list on phones
  var dn = document.querySelector('.doc-nav');
  var dt = document.querySelector('.doc-nav-toggle');
  if (dn && dt) {
    dt.addEventListener('click', function () {
      var open = dn.classList.toggle('open');
      dt.setAttribute('aria-expanded', open ? 'true' : 'false');
    });
  }

  // before / after
  document.querySelectorAll('.compare').forEach(function (c) {
    var input = c.querySelector('input');
    if (!input) return;
    var set = function () { c.style.setProperty('--pos', input.value + '%'); };
    input.addEventListener('input', set);
    set();
  });

  // looping clips play only while on screen (saves battery and bandwidth); honour reduced motion
  var reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  var loops = document.querySelectorAll('video[data-loop]');
  if ('IntersectionObserver' in window) {
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        var v = e.target;
        if (e.isIntersecting && !reduce) {
          if (v.preload === 'none') v.preload = 'auto';
          var p = v.play();
          if (p && p.catch) p.catch(function () {});
        } else {
          v.pause();
        }
      });
    }, { rootMargin: '200px 0px' });
    loops.forEach(function (v) { io.observe(v); });
  }

  // the showreel's sound
  document.querySelectorAll('.sound-btn').forEach(function (b) {
    var v = document.getElementById(b.getAttribute('data-for'));
    if (!v) return;
    b.addEventListener('click', function () {
      if (v.muted) {
        v.muted = false;
        v.currentTime = 0;
        var p = v.play();
        if (p && p.catch) p.catch(function () {});
        b.textContent = 'Sound off';
      } else {
        v.muted = true;
        b.textContent = 'Play with sound';
      }
    });
  });

  // presets: filter chips, and hover (or tap) to play a preset's clip
  var chips = document.querySelectorAll('.filters button');
  chips.forEach(function (chip) {
    chip.addEventListener('click', function () {
      chips.forEach(function (c) { c.setAttribute('aria-pressed', c === chip ? 'true' : 'false'); });
      var cat = chip.getAttribute('data-cat');
      document.querySelectorAll('.preset').forEach(function (p) {
        p.hidden = !(cat === 'all' || p.getAttribute('data-cat') === cat);
      });
    });
  });
  document.querySelectorAll('.preset').forEach(function (p) {
    var v = p.querySelector('video');
    if (!v) return;
    var start = function () { p.classList.add('playing'); v.preload = 'auto'; var r = v.play(); if (r && r.catch) r.catch(function () {}); };
    var stop = function () { p.classList.remove('playing'); v.pause(); };
    p.addEventListener('mouseenter', start);
    p.addEventListener('mouseleave', stop);
    p.addEventListener('focusin', start);
    p.addEventListener('focusout', stop);
  });

  // copy buttons on code blocks
  document.querySelectorAll('pre').forEach(function (pre) {
    if (!navigator.clipboard) return;
    var b = document.createElement('button');
    b.className = 'copy';
    b.type = 'button';
    b.textContent = 'Copy';
    b.addEventListener('click', function () {
      navigator.clipboard.writeText(pre.innerText.replace(/\nCopy$/, '')).then(function () {
        b.textContent = 'Copied';
        setTimeout(function () { b.textContent = 'Copy'; }, 1400);
      });
    });
    pre.appendChild(b);
  });

  // highlight the section being read in the guide's contents
  var links = document.querySelectorAll('.toc a');
  if (links.length && 'IntersectionObserver' in window) {
    var map = {};
    links.forEach(function (a) { map[decodeURIComponent(a.getAttribute('href').slice(1))] = a; });
    var heads = document.querySelectorAll('.prose h2[id], .prose h3[id]');
    var seen = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        if (e.isIntersecting && map[e.target.id]) {
          links.forEach(function (a) { a.classList.remove('active'); });
          map[e.target.id].classList.add('active');
        }
      });
    }, { rootMargin: '-80px 0px -70% 0px' });
    heads.forEach(function (h) { seen.observe(h); });
  }
})();
