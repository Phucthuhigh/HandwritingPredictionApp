//magic.js
//Obtain the canvas and its 2d rendering context
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

//Get the refernce to HTML elements
const brushSize = document.getElementById('brush-size');
const colorPicker = document.getElementById('color-picker');
const clearCanvas = document.getElementById('clear-canvas');
const saveCanvas = document.getElementById('save-canvas');
const predictCanvas = document.getElementById('predict-canvas');
const inputLabel = document.getElementById("label");
const alertBox = document.getElementById("alertBox");

let isDrawing = false;
let isErase = false;

//Initializing the canvas
ctx.fillStyle = "#ffffff";
ctx.fillRect(0, 0, canvas.width, canvas.height);
ctx.lineWidth = 5;
ctx.lineCap = 'round';
ctx.strokeStyle = 'black';

//start drawing
function startPosition(e) {
	isDrawing = true;
	draw(e);
}

//end drawing
function endPosition() {
	isDrawing = false;
	ctx.beginPath();
}

//Function to draw on the Canvas
function draw(e) {
	if (!isDrawing) return;
	if (isErase) ctx.strokeStyle = "#ffffff";
	else ctx.strokeStyle = colorPicker.value; //pick the color
	ctx.lineWidth = brushSize.value; //Select the brush size
	ctx.lineTo(
		e.clientX - canvas.offsetLeft,
		e.clientY - canvas.offsetTop
	);
	ctx.stroke();
	ctx.beginPath();
	ctx.moveTo(
		e.clientX - canvas.offsetLeft,
		e.clientY - canvas.offsetTop
	);
}

function downloadImage(data, filename = 'untitled.jpeg') {
    let a = document.createElement('a');
    a.href = data;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
}

//event listener for differnt mouse actions
canvas.addEventListener('mousedown', startPosition);
canvas.addEventListener('mouseup', endPosition);
canvas.addEventListener('mousemove', draw);
clearCanvas.addEventListener('click', () => {
    ctx.clearRect(
        0, 0, canvas.width,
        canvas.height
    );
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
});

brushSize.addEventListener('input', () => {
	ctx.lineWidth = brushSize.value;
	updateBrushSizeLabel(brushSize.value);
});

function updateBrushSizeLabel(size) {
	const brushSizeLabel = document.getElementById('brush-size-label');
	if (brushSizeLabel) {
		brushSizeLabel.textContent = `Brush Size: ${size}`; 
	}
}

//Get references to the pen and eraser button
const penButton = document.getElementById('pen');
const eraserButton = document.getElementById('eraser');

//switing to pen mode
function activatePen() {
	isErase = false;
	ctx.strokeStyle = colorPicker.value;
}

//switching to eraser mode
function activateEraser() {
	isErase = true;
	ctx.strokeStyle = "#ffffff";
}

function handleAlert(success, message) {
	alertBox.className = success ? "text-green-500" : "text-red-500";
	alertBox.innerHTML = message;
	setTimeout(() => {
		alertBox.className = "";
		alertBox.innerHTML = "";
	}, 2000);
}

penButton.addEventListener('click', () => {
	activatePen();
});

eraserButton.addEventListener('click', () => {
	activateEraser();
});

saveCanvas.addEventListener('click', async () => {
    let imgURL = canvas.toDataURL();
	if (inputLabel.value == "") {
		handleAlert(false, "Label field is required");
		return;
	}
	try {
		const res = await fetch("/save", {
			method: "POST",
			headers: {
				'Content-Type': 'application/json'
			},
			body: JSON.stringify({label: inputLabel.value, image_data: imgURL})
		})
		const data = await res.json();
		console.log(data.success);
		if (data.success) {
			handleAlert(data.success, "Save successfully!");
		} else {
			handleAlert(data.success, "Server error:(");
		}
	} catch (error) {
		console.log(error);
	}
    // downloadImage(imgURL, "my-canvas.jpeg");
});

predictCanvas.addEventListener('click', async () => {
	let imgURL = canvas.toDataURL();
	try {
		const res = await fetch("/predict", {
			method: "POST",
			headers: {
				'Content-Type': 'application/json'
			},
			body: JSON.stringify({image_data: imgURL})
		})
		const data = await res.json();
	} catch (error) {
		console.log(error);
	}
});