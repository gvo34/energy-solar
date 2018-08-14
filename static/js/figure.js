var figure = {
    "frames": [], 
    "layout": {
        "autosize": true, 
        "undefined": {
            "rangeslider": {
                "visible": true
            }
        }, 
        "title": "Exploration into Monthly Solar Energy using four distinct techniques:", 
        "yaxis": {
            "range": [
                1.6348967222222222, 
                6.816302277777778
            ], 
            "type": "linear", 
            "autorange": true, 
            "title": "StandardScaler trillion BTU"
        }, 
        "showlegend": true, 
        "breakpoints": [], 
        "xaxis": {
            "range": [
                0, 
                11
            ], 
            "title": "<br>", 
            "type": "category", 
            "autorange": true, 
            "rangeslider": {
                "visible": true, 
                "range": [
                    0, 
                    11
                ], 
                "autorange": true, 
                "yaxis": {
                    "rangemode": "match"
                }
            }
        }, 
        "hovermode": "closest"
    }, 
    "data": [
        {
            "autobinx": true, 
            "name": "Solar Energy", 
            "yaxis": "y", 
            "ysrc": "manuelamachado:12:ec39ac", 
            "xsrc": "manuelamachado:12:76f9a4", 
            "marker": {
                "color": "rgb(160, 180, 31)"
            }, 
            "mode": "lines", 
            "xaxis": "x", 
            "hoverinfo": "x+y+name", 
            "y": [
                "4.595387", 
                "5.403802", 
                "5.684115", 
                "6.034118", 
                "5.773938", 
                "4.832852", 
                "4.020823", 
                "2.761224", 
                "2.259449", 
                "2.143813", 
                "2.739026", 
                "5.282488"
            ], 
            "x": [
                "April - 2016", 
                "May - 2016", 
                "June - 2016", 
                "July - 2016", 
                "August - 2016", 
                "September - 2016", 
                "October - 2016", 
                "November - 2016", 
                "December - 2016", 
                "January - 2017", 
                "February - 2017", 
                "March - 2017"
            ], 
            "line": {
                "dash": "solid", 
                "color": "rgb(8, 90, 144)"
            }, 
            "type": "timeseries", 
            "autobiny": true
        }, 
        {
            "name": "MLP prediction", 
            "ysrc": "manuelamachado:12:0cce70", 
            "xsrc": "manuelamachado:12:76f9a4", 
            "marker": {
                "color": "rgb(23, 190, 207)", 
                "opacity": 1, 
                "symbol": "hexagon2"
            }, 
            "mode": "lines", 
            "hoverinfo": "x+y+name", 
            "y": [
                "4.998103", 
                "5.662354", 
                "6.205909", 
                "6.510017", 
                "6.557232", 
                "5.685867", 
                "4.315407", 
                "3.492486", 
                "2.718280", 
                "2.997740", 
                "4.082729", 
                "5.015638"
            ], 
            "x": [
                "April - 2016", 
                "May - 2016", 
                "June - 2016", 
                "July - 2016", 
                "August - 2016", 
                "September - 2016", 
                "October - 2016", 
                "November - 2016", 
                "December - 2016", 
                "January - 2017", 
                "February - 2017", 
                "March - 2017"
            ], 
            "line": {
                "dash": "dot"
            }, 
            "type": "timeseries"
        }, 
        {
            "name": "LR prediction", 
            "ysrc": "manuelamachado:12:c5b036", 
            "xsrc": "manuelamachado:12:76f9a4", 
            "mode": "lines", 
            "y": [
                "4.749506", 
                "5.194101", 
                "5.930128", 
                "5.68767", 
                "6.036923", 
                "4.53533", 
                "4.192624", 
                "2.590346", 
                "2.415662", 
                "1.893967", 
                "3.292849", 
                "4.693661"
            ], 
            "x": [
                "April - 2016", 
                "May - 2016", 
                "June - 2016", 
                "July - 2016", 
                "August - 2016", 
                "September - 2016", 
                "October - 2016", 
                "November - 2016", 
                "December - 2016", 
                "January - 2017", 
                "February - 2017", 
                "March - 2017"
            ], 
            "line": {
                "dash": "dash"
            }, 
            "type": "timeseries"
        }, 
        {
            "name": "AR prediction", 
            "ysrc": "manuelamachado:12:6b03e3", 
            "xsrc": "manuelamachado:12:76f9a4", 
            "mode": "lines", 
            "hoverinfo": "x+y+name", 
            "y": [
                "4.931477516", 
                "5.675513164", 
                "5.854505775", 
                "6.050279168", 
                "5.783258739", 
                "4.849464476", 
                "4.21476772", 
                "3.067250411", 
                "2.356242538", 
                "2.356464788", 
                "3.304432299", 
                "5.210101608"
            ], 
            "x": [
                "April - 2016", 
                "May - 2016", 
                "June - 2016", 
                "July - 2016", 
                "August - 2016", 
                "September - 2016", 
                "October - 2016", 
                "November - 2016", 
                "December - 2016", 
                "January - 2017", 
                "February - 2017", 
                "March - 2017"
            ], 
            "line": {
                "dash": "dashdot", 
                "color": "rgb(255, 127, 14)"
            }, 
            "type": "timeseries"
        }, 
        {
            "name": "RF prediction", 
            "ysrc": "manuelamachado:12:d8f27e", 
            "xsrc": "manuelamachado:12:76f9a4", 
            "mode": "lines", 
            "y": [
                "4.937626", 
                "5.516971", 
                "5.725167", 
                "5.935274", 
                "5.796133", 
                "4.942688", 
                "3.843808", 
                "2.692608", 
                "2.331841", 
                "2.273994", 
                "2.738833", 
                "5.075115"
            ], 
            "x": [
                "April - 2016", 
                "May - 2016", 
                "June - 2016", 
                "July - 2016", 
                "August - 2016", 
                "September - 2016", 
                "October - 2016", 
                "November - 2016", 
                "December - 2016", 
                "January - 2017", 
                "February - 2017", 
                "March - 2017"
            ], 
            "line": {
                "dash": "dot", 
                "color": "rgb(200, 213, 33)"
            }, 
            "type": "timeseries"
        }
    ]
}