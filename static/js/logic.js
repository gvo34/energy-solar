 
var init_value = document.getElementById('thedropdown');

console.log(init_value.options[init_value.selectedIndex].value)



  

function ARHistory(history){
    console.log("calling Autoregression with history ", history)
    d3.json(`/ARHistory/${history}`, (error, response) => {
        if (error) return console.warn(error);
        
        // Model score
        console.log("ARH with history of ", history, " score of ", response);
        
        // Update prediction score
        var init_score = document.getElementById('prediction_ARH');
        init_score.innerHTML = "MSE: "+ response.MSE;

        //refresh image NEED THIS TO WORK
        //document.getElementById('ARimg').innerHTML = "static/images/AR_residual.png"
      });
}

function Autoregression(history){
    console.log("calling Autoregression ", history)
    d3.json(`/Autoregression/${history}`, (error, response) => {
        if (error) return console.warn(error);
        
        // Model score
        console.log("AR with history of ", history, " score of ", response);
        
        // Update prediction score
        var init_score = document.getElementById('prediction_AR');
        init_score.innerHTML = "MSE: " + response.MSE;

        //refresh image NEED THIS TO WORK
        //document.getElementById('ARimg').innerHTML = "static/images/AR_residual.png"
      });
}


function Linear(history){
    console.log("calling Linear Regression with history ", history)
    d3.json(`/Linear/${history}`, (error, response) => {
        if (error) return console.warn(error);
        
        // Model score
        console.log("LR with the history of ", history, " score of ", response);
        
        // Update prediction score
        var init_score = document.getElementById('prediction_LR');
        init_score.innerHTML = "MSE: "+response.MSE + " r2: "+ response.r2;
    });
}

function MLP(history){
    console.log("calling MLP with history ", history)
    d3.json(`/MLP/${history}`, (error, response) => {
        if (error) return console.warn(error);
        
        // Model score
        console.log("MLP with the history of ", history, " score of ", response);
        
        // Update prediction score
        var init_score = document.getElementById('prediction_MLP');
        init_score.innerHTML = "MSE: "+response.MSE + " r2: "+ response.r2;

        //refresh image NEED THIS TO WORK
        //document.getElementById('LRimg').innerHTML = "static/images/LR_residual.png"
      });
}

function RandomForrest(history){
    console.log("calling RandomForrest with history ", history)
    d3.json(`/RF/${history}`, (error, response) => {
        if (error) return console.warn(error);
        
        // Model score
        console.log("RF with the history of ", history, " score of ", response);
        
        // Update prediction score
        var init_score = document.getElementById('prediction_RF');
        init_score.innerHTML = "MSE: "+response.MSE + " r2: "+ response.r2;

        //refresh image NEED THIS TO WORK
        //document.getElementById('LRimg').innerHTML = "static/images/LR_residual.png"
      });
}

function RFfuzzy(history){
    console.log("calling RandomForrest Fuzzy with history ", history)
    d3.json(`/RFfuzzy/${history}`, (error, response) => {
        if (error) return console.warn(error);
        
        // Model score
        console.log("RF Fuzzy with the history of ", history, " score of ", response);
        
        // Update prediction score
        var init_score = document.getElementById('prediction_RF');
        init_score.innerHTML = "MSE: "+response.MSE + " r2: "+ response.r2;

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
    MLP(newvalue);
    //RandomForrest(newvalue);
    RFfuzzy(newvalue);


}

