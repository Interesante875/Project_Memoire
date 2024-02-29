// This is the JavaScript code

function getCursor(){
    const cursor = document.querySelector("#cursor");

    // cursor.style.border = '2px solid #E61014'
    // Get the cursor's size
    const cursorWidth = cursor.offsetWidth;
    const cursorHeight = cursor.offsetHeight;

    // Add a "clicked" class to the cursor when the user clicks the mouse
    document.addEventListener("click", () => {
        cursor.classList.add("clicked");
        // Remove the "clicked" class after the pulse animation has completed
        setTimeout(() => {
            cursor.classList.remove("clicked");
        }, 1000);
    });

    // Follow the mouse movement
    document.addEventListener("mousemove", (event) => {
        // cursor.style.top = event.pageY + "px";
        // cursor.style.left = event.pageX + "px";
        // Set the top and left values to center the cursor above the mouse
        cursor.style.top = event.pageY - (cursorHeight / 2) + "px";
        cursor.style.left = event.pageX - (cursorWidth / 2) + "px";
    });
}


document.getElementById('title-page').disabled  = false;
document.getElementById('demo-page').disabled = true;
getCursor();
// dragElement(document.getElementsByClassName("btn__bookmark"));

// function dragElement(elmnt) {
//     var pos1 = 0, pos2 = 0, pos3 = 0, pos4 = 0;
//     if (document.getElementById(elmnt.id + "header")) {
//         /* if present, the header is where you move the DIV from:*/
//         document.getElementById(elmnt.id + "header").onmousedown = dragMouseDown;
//     } else {
//         /* otherwise, move the DIV from anywhere inside the DIV:*/
//         elmnt.onmousedown = dragMouseDown;
//     }

//     function dragMouseDown(e) {
//         e = e || window.event;
//         e.preventDefault();
//         // get the mouse cursor position at startup:
//         pos3 = e.clientX;
//         pos4 = e.clientY;
//         document.onmouseup = closeDragElement;
//         // call a function whenever the cursor moves:
//         document.onmousemove = elementDrag;
//     }

//     function elementDrag(e) {
//         e = e || window.event;
//         e.preventDefault();
//         // calculate the new cursor position:
//         pos1 = pos3 - e.clientX;
//         pos2 = pos4 - e.clientY;
//         pos3 = e.clientX;
//         pos4 = e.clientY;
//         // set the element's new position:
//         elmnt.style.top = (elmnt.offsetTop - pos2) + "px";
//         elmnt.style.left = (elmnt.offsetLeft - pos1) + "px";
//     }

//     function closeDragElement() {
//         /* stop moving when mouse button is released:*/
//         document.onmouseup = null;
//         document.onmousemove = null;
//     }
// }