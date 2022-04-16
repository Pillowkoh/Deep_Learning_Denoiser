$(document).ready(function () {

    const inputbutton = $('#audio-input');
    const inputsrc = $('#input');

    let file;

    inputbutton.change(function () {
        file = this.files[0];
        console.log(file);
        inputsrc.attr("src", URL.createObjectURL(this.files[0]));
        $("#input-display")[0].load();
    });

    const denoisebutton = $('#denoise');

    function runPyFile(f) {
        console.log('working on it,,,')
        $.ajax({
            url: "test.py",
            context: document.body
        })
    }

    denoisebutton.click(function () {
        if (file != undefined) {
            console.log(file);
            console.log("denoise!");
            runPyFile(file);
        } else {
            console.log("no file to denoise!")
        }
    })

})
