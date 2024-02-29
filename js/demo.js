const page = document.getElementById("container");




function getCursor(){
    const cursor = document.querySelector("#cursor");
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

function changeStyle(){

    document.getElementById('title-page').disabled  = true;
    document.getElementById('demo-page').disabled = false;

    renderDemo();
    getCursor();
    addSlider();

}

function renderDemo() {
    page.innerHTML = `
        <div class="background__container">
            <div class="decorator__container">
                <div class="decorator__word"></div>
                <div class="decorator__talent"></div>
                <div class="decorator__star"></div>
                <div class="decorator__umbrella"></div>
                <div class="decorator__logo"></div>
            </div>

            <div class="banner__container">
                <div class="banner__container--character">
                    <img class="character__image" id="sadistic" src='../assets/animated/tassel_aru.gif'>
                    <div class="banner__container--glass"></div>
                </div>

                <div class="banner__container--character" >
                    <img class="character__image" id="nilou" src='../assets/animated/tassel_io.gif'>
                    <div class="banner__container--glass"></div>
                </div>


            </div>
        

            <div class="btn__container">
                <img class="back-btn" id="back" src='../assets/interactive/Backbutton.png'>
                <img class="forw-btn" id="forward" src='../assets/interactive/Forwardbutton.png'>
            </div>

        

        </div>

        <div id="cursor">
            <div id="inner-cursor"></div>
        </div>


    `

}

function addSlider(){
    const slider = document.querySelector('.banner__container');
    console.log(slider);
    let isDown = false;
    let startX;
    let scrollLeft;

    slider.addEventListener('mousedown', (e) => {
    isDown = true;
    slider.classList.add('active');
    startX = e.pageX - slider.offsetLeft;
    scrollLeft = slider.scrollLeft;
    });
    slider.addEventListener('mouseleave', () => {
    isDown = false;
    slider.classList.remove('active');
    });
    slider.addEventListener('mouseup', () => {
    isDown = false;
    slider.classList.remove('active');
    });
    slider.addEventListener('mousemove', (e) => {
    if(!isDown) return;
    e.preventDefault();
    const x = e.pageX - slider.offsetLeft;
    const walk = (x - startX) * 3; //scroll-fast
    slider.scrollLeft = scrollLeft - walk;
    console.log(walk);
    });
}


// function renderDemo() {
//     page.innerHTML = `
//         <div class="background__container">
//             <div class="decorator__container">
//                 <div class="decorator__word"></div>
//                 <div class="decorator__talent"></div>
//                 <div class="decorator__star"></div>
//                 <div class="decorator__umbrella"></div>
//                 <div class="decorator__logo"></div>
//             </div>

//             <div class="banner__container">
//                 <div class="banner__container--character">
//                     <div class="character__image" id="sadistic"></div>
//                     <div class="string__container">
//                         <div class="decorator__string--1"></div>
//                         <div class="decorator__string--2" id="string1"></div>
//                     </div>
//                 </div>

//                 <div class="banner__container--character" >
//                     <div class="character__image" id="nilou"></div>
//                     <div class="string__container">
//                         <div class="decorator__string--1"></div>
//                         <div class="decorator__string--2" id="string2"></div>
//                     </div>
//                 </div>

//             </div>
        
            

//         </div>

//         <div id="cursor">
//             <div id="inner-cursor"></div>
//         </div>


//     `

// }