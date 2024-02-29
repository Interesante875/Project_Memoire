//- Explosion
//- adapted from "Anchor Click Canvas Animation" by Nick Sheffield
//- https://codepen.io/nicksheffield/pen/NNEoLg/
// svg.c-logo.js-explosion(xmlns="http://www.w3.org/2000/svg" viewBox="0 0 500 500" width="300" height="300" fill="currentColor")
//     path(d="M111.7 74.9L91.2 93.1l9.1 10.2 17.8-15.8 7.4 8.4-17.8 15.8 10.1 11.4 20.6-18.2 7.7 8.7-30.4 26.9-41.9-47.3 30.3-26.9 7.6 8.6zM190.8 59.6L219 84.3l-14.4 4.5-20.4-18.2-6.4 26.6-14.4 4.5 8.9-36.4-26.9-24.1 14.3-4.5L179 54.2l5.7-25.2 14.3-4.5-8.2 35.1zM250.1 21.2l27.1 3.4c6.1.8 10.8 3.1 14 7.2 3.2 4.1 4.5 9.2 3.7 15.5-.8 6.3-3.2 11-7.4 14.1-4.1 3.1-9.2 4.3-15.3 3.5L258 63.2l-2.8 22.3-13-1.6 7.9-62.7zm11.5 13l-2.2 17.5 12.6 1.6c5.1.6 9.1-2 9.8-7.6.7-5.6-2.5-9.2-7.6-9.9l-12.6-1.6zM329.1 95.4l23.8 13.8-5.8 10L312 98.8l31.8-54.6 11.3 6.6-26 44.6zM440.5 145c-1.3 8.4-5.9 15.4-13.9 21.1s-16.2 7.7-24.6 6.1c-8.4-1.6-15.3-6.3-20.8-14.1-5.5-7.9-7.6-16-6.4-24.4 1.3-8.5 6-15.5 14-21.1 8-5.6 16.2-7.7 24.5-6 8.4 1.6 15.4 6.3 20.9 14.2 5.5 7.6 7.6 15.7 6.3 24.2zM412 119c-5.1-.8-10.3.6-15.6 4.4-5.2 3.7-8.4 8.1-9.4 13.2-1 5.2.2 10.1 3.5 14.8 3.4 4.8 7.5 7.5 12.7 8.2 5.2.8 10.4-.7 15.6-4.4 5.3-3.7 8.4-8.1 9.4-13.2 1.1-5.1-.1-9.9-3.4-14.7-3.4-4.8-7.6-7.6-12.8-8.3zM471.5 237.9c-2.8 4.8-7.1 7.6-13 8.7l-2.6-13.1c5.3-.9 8.1-5 7.2-11-.9-5.8-4.3-8.8-8.9-8.2-2.3.3-3.7 1.4-4.5 3.3-.7 1.9-1.4 5.2-1.7 10.1-.8 7.5-2.2 13.1-4.3 16.9-2.1 3.9-5.7 6.2-10.9 7-6.3.9-11.3-.5-15.2-4.4-3.9-3.8-6.3-9-7.3-15.7-1.1-7.4-.2-13.7 2.6-18.8 2.8-5.1 7.4-8.2 13.7-9.2l2.6 13c-5.6 1.1-8.7 6.6-7.7 13.4 1 6.6 3.9 9.5 8.6 8.8 4.4-.7 5.7-4.5 6.7-14.1.3-3.5.7-6.2 1.1-8.4.4-2.2 1.2-4.4 2.2-6.8 2.1-4.7 6-7.2 11.8-8.1 5.4-.8 10.3.4 14.5 3.7 4.2 3.3 6.9 8.5 8 15.6.9 6.9-.1 12.6-2.9 17.3zM408.6 293.5l2.4-12.9 62 11.7-2.4 12.9-62-11.7zM419.6 396.9c-8.3 2-16.5.3-24.8-5-8.2-5.3-13.2-12.1-14.9-20.5-1.6-8.4.1-16.6 5.3-24.6 5.2-8.1 11.9-13.1 20.2-15.1 8.4-1.9 16.6-.3 24.9 5 8.2 5.3 13.2 12.1 14.8 20.5 1.7 8.4 0 16.6-5.2 24.7-5.2 8-12 13-20.3 15zm13.4-36.3c-1.2-5.1-4.5-9.3-9.9-12.8s-10.6-4.7-15.8-3.7-9.3 4-12.4 8.9-4.1 9.8-2.8 14.8c1.2 5.1 4.5 9.3 9.9 12.8 5.5 3.5 10.7 4.8 15.8 3.7 5.1-.9 9.2-3.8 12.3-8.7s4.1-9.9 2.9-15zM303.6 416.5l9.6-5.4 43.3 20.4-19.2-34 11.4-6.4 31 55-9.6 5.4-43.4-20.5 19.2 34.1-11.3 6.4-31-55zM238.2 468.8c-49 0-96.9-17.4-134.8-49-38.3-32-64-76.7-72.5-125.9-2-11.9-3.1-24-3.1-35.9 0-36.5 9.6-72.6 27.9-104.4 2.1-3.6 6.7-4.9 10.3-2.8 3.6 2.1 4.9 6.7 2.8 10.3-16.9 29.5-25.9 63.1-25.9 96.9 0 11.1 1 22.3 2.9 33.4 7.9 45.7 31.8 87.2 67.3 116.9 35.2 29.3 79.6 45.5 125.1 45.5 11.1 0 22.3-1 33.4-2.9 4.1-.7 8 2 8.7 6.1.7 4.1-2 8-6.1 8.7-11.9 2-24 3.1-36 3.1z")
{
    const colors = [ '#ffc000', '#ff3b3b', '#ff8400' ];
    const bubbles = 1;

    const explode = (x, y) => {
        let particles = [];
        let ratio = window.devicePixelRatio;
        let c = document.createElement('canvas');
        let ctx = c.getContext('2d');

        c.style.position = 'absolute';
        c.style.left = (x - 100) + 'px';
        c.style.top = (y - 100) + 'px';
        c.style.pointerEvents = 'none';
        c.style.width = 200 + 'px';
        c.style.height = 200 + 'px';
        
        c.style.zIndex = 100;
        c.width = 200 * ratio;
        c.height = 200 * ratio;
        document.body.appendChild(c);

        for(var i = 0; i < bubbles; i++) {
            particles.push({
                x: c.width / 2,
                y: c.height / 2,
                radius: r(20, 30),
                color: colors[Math.floor(Math.random() * colors.length)],
                rotation: r(0, 360, true),
                speed: r(8, 12),
                friction: 0.9,
                opacity: r(0, 0.5, true),
                yVel: 0,
                gravity: 0.1
            });
        }

        render(particles, ctx, c.width, c.height);
        setTimeout(() => document.body.removeChild(c), 250);
    }

    const render = (particles, ctx, width, height) => {
        requestAnimationFrame(() => render(particles, ctx, width, height));
        ctx.clearRect(0, 0, width, height);

        particles.forEach((p, i) => {
            p.x += p.speed * Math.cos(p.rotation * Math.PI / 180);
            p.y += p.speed * Math.sin(p.rotation * Math.PI / 180);

            p.opacity -= 0.01;
            p.speed *= p.friction;
            p.radius *= p.friction;
            p.yVel += p.gravity;
            p.y += p.yVel;

            if(p.opacity < 0 || p.radius < 0) return;

            ctx.beginPath();
            ctx.globalAlpha = p.opacity;
            ctx.fillStyle = p.color;
            ctx.arc(p.x, p.y, p.radius, 0, 2 * Math.PI, false);
            ctx.fill();
        });

        return ctx;
    }

    const r = (a, b, c) => parseFloat((Math.random() * ((a ? a : 1) - (b ? b : 0)) + (b ? b : 0)).toFixed(c ? c : 0));

    document.querySelector('body').addEventListener('click', e => explode(e.pageX, e.pageY));
}