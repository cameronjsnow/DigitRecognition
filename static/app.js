const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
let drawing = false;

// Set up canvas properties
ctx.lineWidth = 20;
ctx.lineCap = 'round';
ctx.strokeStyle = 'white';

// Mouse events
canvas.addEventListener('mousedown', (e) => {
    drawing = true;
    draw(e);
});

canvas.addEventListener('mousemove', draw);

canvas.addEventListener('mouseup', () => {
    drawing = false;
    ctx.beginPath();
    sendImage();
});

canvas.addEventListener('mouseout', () => {
    drawing = false;
    ctx.beginPath();
});

// Touch events for mobile devices
canvas.addEventListener('touchstart', (e) => {
    e.preventDefault();
    drawing = true;
    drawTouch(e);
});

canvas.addEventListener('touchmove', drawTouch);

canvas.addEventListener('touchend', () => {
    drawing = false;
    ctx.beginPath();
    sendImage();
});

function draw(e) {
    if (!drawing) return;
    ctx.lineTo(e.offsetX, e.offsetY);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(e.offsetX, e.offsetY);
}

function drawTouch(e) {
    if (!drawing) return;
    var touch = e.touches[0];
    var rect = canvas.getBoundingClientRect();
    var x = touch.clientX - rect.left;
    var y = touch.clientY - rect.top;

    ctx.lineTo(x, y);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x, y);
}

document.getElementById('clear-btn').addEventListener('click', () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    document.getElementById('prediction').innerText = '';
    document.getElementById('top-predictions').innerHTML = '';
});

function sendImage() {
    const dataURL = canvas.toDataURL('image/png');
    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ image: dataURL })
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('prediction').innerText = data.top_prediction || data.prediction;
        if (data.predictions) {
            let html = '<h3 class="text-lg font-semibold mb-2">Top Predictions:</h3><ul class="list-disc list-inside">';
            data.predictions.forEach(pred => {
                html += `<li>Digit: ${pred.digit}, Prob: ${pred.probability}</li>`;
            });
            html += '</ul>';
            document.getElementById('top-predictions').innerHTML = html;
        }
    })
    .catch(error => console.error('Error:', error));
}

// Help Overlay Logic
const helpBtn = document.getElementById('help-btn');
const helpOverlay = document.getElementById('help-overlay');
const closeHelp = document.getElementById('close-help');

helpBtn.addEventListener('click', () => {
    helpOverlay.classList.remove('hidden');
});

closeHelp.addEventListener('click', () => {
    helpOverlay.classList.add('hidden');
});
