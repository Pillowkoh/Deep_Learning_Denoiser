$(document).ready(function () {

    const inputbutton = $('#audio-input');
    const inputsrc = $('#input');
    
    inputbutton.change(function() {
        console.log(this.files[0])
        inputsrc.attr("src", URL.createObjectURL(this.files[0]));
        $("#input-display")[0].load();
    })

    const denoisebutton = $('#denoise');

    denoisebutton.click( function() {
        let file = inputbutton.files[0];
        if (file != undefined) {
            console.log("denoise!");
        } else {
            console.log("no file to denoise!")
        }
    })

})
