// userId = '1915Hall';

// // Make a request for a user with a given ID
// axios.get('http://ec2-3-18-9-170.us-east-2.compute.amazonaws.com/get_past_consumption/' + userId, { crossdomain: true })
// 	.then(function (response) {
// 	// handle success
// 	console.log(response);
// 	})
// 	.catch(function (error) {
// 	// handle error
// 	console.log(error);
// 	})

function httpGetAsync(theUrl, callback)
{
    var xmlHttp = new XMLHttpRequest();
    xmlHttp.onreadystatechange = function() {
        if (xmlHttp.readyState == 4 && xmlHttp.status == 200)
            callback(xmlHttp.responseText);
    }
    xmlHttp.open("GET", theUrl, true); // true for asynchronous
    xmlHttp.send(null);
}

httpGetAsync('./api-past-consumption-1915Hall', function(response) {
	console.log(response);
});


Plotly.d3.csv("https://raw.githubusercontent.com/sysoce/junction/master/data/buildings-2018-3.csv", function(err, rows){
	function unpack(rows, key) {
		return rows.map(function(row) { return row[key]; });
	}

	var trace1 = {
	  type: "scatter",
	  mode: "lines",
	  name: '1927Hall',
	  x: unpack(rows, 'Time (UTC)'),
	  y: unpack(rows, '1927Hall.kW'),
	  line: {color: '#17BECF'}
	}

	var trace2 = {
	  type: "scatter",
	  mode: "lines",
	  name: '1937Feinberg',
	  x: unpack(rows, 'Time (UTC)'),
	  y: unpack(rows, '1937Feinberg.kW'),
	  line: {color: '#7F7F7F'}
	}

	var data = [trace1,trace2];

	var layout = {
	  title: 'Consumption',
	};

	Plotly.newPlot('myDiv', data, layout, {showSendToCloud: true});
});