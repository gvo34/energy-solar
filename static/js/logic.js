// Creating our initial map object
// We set the longitude, latitude, and the starting zoom level
// This gets inserted into the div with an id of 'map'
var myMap = L.map('mapit', {
    center: [36.77, -119.41],
    zoom: 5,
  });
  
  // Adding a tile layer (the background map image) to our map
  // We use the addTo method to add objects to our map
  var access_token =
    'pk.eyJ1IjoiZ3Vpcmx5biIsImEiOiJjamh0dzZnaHowaTlnM3BvNGl3NzYwNDQ2In0.fjmZrTxDywSwzCeE6BYKUg';
  L.tileLayer(
    'https://api.mapbox.com/styles/v1/mapbox/outdoors-v10/tiles/256/{z}/{x}/{y}?' +
      'access_token=' +
      access_token,
  ).addTo(myMap);
  
