 
var init_value = document.getElementById('thedropdown');

console.log(init_value.options[init_value.selectedIndex].value)


function ARHistory(history){
    console.log("calling Autoregression with history ", history)
    d3.json(`/Autoregression/${history}`, (error, response) => {
        if (error) return console.warn(error);
        
        // Model score
        console.log("pass the history of ", history, " Obtained model score of ", response);
        
        // Update prediction score
        var init_score = document.getElementById('prediction_ARH');
        init_score.innerHTML = response.MSE;

        //refresh image NEED THIS TO WORK
        //document.getElementById('ARimg').innerHTML = "static/images/AR_residual.png"
      });
}

function Autoregression(history){
    console.log("calling Autoregression ", history)
    d3.json(`/Autoregression/${history}`, (error, response) => {
        if (error) return console.warn(error);
        
        // Model score
        console.log("pass the history of ", history, " Obtained model score of ", response);
        
        // Update prediction score
        var init_score = document.getElementById('prediction_AR');
        init_score.innerHTML = response.MSE;

        //refresh image NEED THIS TO WORK
        //document.getElementById('ARimg').innerHTML = "static/images/AR_residual.png"
      });
}




function Linear(history){
    console.log("calling Linear Regression with history ", history)
    d3.json(`/Linear/${history}`, (error, response) => {
        if (error) return console.warn(error);
        
        // Model score
        console.log("pass the history of ", history, " Obtained model score of ", response);
        
        // Update prediction score
        var init_score = document.getElementById('prediction_LR');
        init_score.innerHTML = response.r2;

        //refresh image NEED THIS TO WORK
        //document.getElementById('LRimg').innerHTML = "static/images/LR_residual.png"
      });
}

function gethistory(live_value){

    newvalue = live_value.options[live_value.selectedIndex].value;
    console.log("value changed to ",newvalue );

    Linear(newvalue);
    Autoregression(newvalue);
    ARHistory(newvalue);

}

