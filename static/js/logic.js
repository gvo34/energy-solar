// Creating our initial map object
// We set the longitude, latitude, and the starting zoom level
// This gets inserted into the div with an id of 'map'
// var myMap = L.map('mapit', {
//     center: [36.77, -119.41],
//     zoom: 5,
//   });
  
//   // Adding a tile layer (the background map image) to our map
//   // We use the addTo method to add objects to our map
//   var access_token =
//     'pk.eyJ1IjoiZ3Vpcmx5biIsImEiOiJjamh0dzZnaHowaTlnM3BvNGl3NzYwNDQ2In0.fjmZrTxDywSwzCeE6BYKUg';
//   L.tileLayer(
//     'https://api.mapbox.com/styles/v1/mapbox/outdoors-v10/tiles/256/{z}/{x}/{y}?' +
//       'access_token=' +
//       access_token,
//   ).addTo(myMap);
  
// var history = document.getElementById("#thedropdown");
// console.log(history);

// var value = history.options[history.selectedIndex].value;

//var value = d3.select("#thedropdown").node().value;
var init_value = document.getElementById('thedropdown');
console.log(init_value.options[init_value.selectedIndex].value)


function Linear(history){
    console.log("calling with history ", history)
    d3.json(`/Linear/${history}`, (error, response) => {
        if (error) return console.warn(error);
        
        // draw gauge based on example table loaded in response
        console.log("pass the history of ", history)
      });
}

function gethistory(live_value){

    newvalue = live_value.options[live_value.selectedIndex].value;
    console.log("value changed to ",newvalue );

    Linear(newvalue);

}

