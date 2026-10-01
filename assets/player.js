(() => {
  'use strict';
  const allAudio = [...document.querySelectorAll('audio')];
  const status = document.querySelector('#playback-status');
  const formatTime = value => `${Math.floor(value / 60)}:${String(Math.floor(value % 60)).padStart(2, '0')}`;
  for (const audio of allAudio) {
    const wrap = audio.closest('.audio-wrap');
    const card = audio.closest('.method');
    const title = audio.dataset.label;
    const player = document.createElement('div');
    player.className = 'player';
    player.innerHTML = `<button class="play-button" type="button" aria-pressed="false"><svg class="play-icon" viewBox="0 0 12 12" aria-hidden="true"><path d="M3 1.5 11 6 3 10.5Z"/></svg><svg class="pause-icon" viewBox="0 0 12 12" aria-hidden="true"><path d="M2 1h3v10H2zm5 0h3v10H7z"/></svg></button><input class="seek" type="range" min="0" step="0.01" value="0">`;
    const button = player.querySelector('button');
    const seek = player.querySelector('input');
    const clock = wrap.querySelector('.time');
    let duration = Number(audio.dataset.duration);
    seek.max = duration;
    seek.setAttribute('aria-label', `Seek ${title}`);
    button.setAttribute('aria-label', `Play ${title}`);
    const updateClock = () => {
      const current = Number.isFinite(audio.currentTime) ? audio.currentTime : 0;
      seek.value = current;
      clock.textContent = formatTime(current);
      seek.setAttribute('aria-valuetext', `${formatTime(current)} of ${formatTime(duration)}`);
    };
    const setPlaying = playing => {
      button.setAttribute('aria-pressed', String(playing));
      button.setAttribute('aria-label', `${playing ? 'Pause' : 'Play'} ${title}`);
      card.classList.toggle('playing', playing);
    };
    button.addEventListener('click', async () => {
      if (!audio.paused) { audio.pause(); return; }
      allAudio.forEach(other => { if (other !== audio) other.pause(); });
      try { await audio.play(); } catch (error) {
        if (error.name === 'AbortError') return;
        status.textContent = `Unable to play ${title}. Please try again.`;
        button.title = 'Unable to play. Please try again.';
      }
    });
    audio.addEventListener('play', () => {
      allAudio.forEach(other => { if (other !== audio) other.pause(); });
      setPlaying(true);
      status.textContent = `Playing ${title}`;
    });
    audio.addEventListener('pause', () => setPlaying(false));
    audio.addEventListener('ended', () => { setPlaying(false); updateClock(); });
    audio.addEventListener('loadedmetadata', () => {
      if (Number.isFinite(audio.duration)) { duration = audio.duration; seek.max = duration; }
      updateClock();
    });
    audio.addEventListener('timeupdate', updateClock);
    audio.addEventListener('error', () => { setPlaying(false); status.textContent = `Unable to load ${title}. Please reload the page and try again.`; });
    seek.addEventListener('input', () => { audio.currentTime = Math.min(Number(seek.value), duration); updateClock(); });
    wrap.insertBefore(player, wrap.querySelector('.audio-bottom'));
    wrap.classList.add('enhanced');
    updateClock();
  }
  const dialog = document.querySelector('#spectrogram-dialog');
  if (dialog && typeof dialog.showModal === 'function') {
    document.querySelectorAll('.plot').forEach(link => link.addEventListener('click', event => {
      if (event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
      event.preventDefault();
      document.querySelector('#dialog-title').textContent = link.dataset.title;
      const img = dialog.querySelector('.dialog-image');
      img.src = link.href;
      img.alt = link.querySelector('img').alt;
      dialog.querySelector('.dialog-original').href = link.href;
      dialog.showModal();
    }));
    dialog.querySelector('.dialog-close').addEventListener('click', () => dialog.close());
    dialog.addEventListener('click', event => { if (event.target === dialog) { const r = dialog.getBoundingClientRect(); if (event.clientX < r.left || event.clientX > r.right || event.clientY < r.top || event.clientY > r.bottom) dialog.close(); } });
  }
  const navLinks = [...document.querySelectorAll('.nav-links a')];
  if ('IntersectionObserver' in window) {
    const observer = new IntersectionObserver(entries => {
      const visible = entries.filter(entry => entry.isIntersecting).sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
      if (!visible) return;
      navLinks.forEach(link => {
        const active = link.hash === `#${visible.target.id}`;
        link.classList.toggle('active', active);
        if (active) link.setAttribute('aria-current', 'location'); else link.removeAttribute('aria-current');
      });
    }, { rootMargin: '-90px 0px -40% 0px', threshold: [0, .2, .5] });
    document.querySelectorAll('.sample').forEach(sample => observer.observe(sample));
  }
})();
