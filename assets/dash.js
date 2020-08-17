var cache = {}
function setThumbnail(row) {
	return new Promise(function(resolve, reject){
		var video = document.getElementById('hidden-media');
		var canvas = document.getElementById('hidden-canvas');
		var ctx = canvas.getContext('2d');
		canvas.width = video.videoWidth;
		canvas.height = video.videoHeight;
		video.muted = true;
	
		let channel = row.querySelector('td[data-dash-column="Channel"]').textContent
	
		if (channel == 'Video') {
	
			let frameIdx = parseInt(row.querySelector('td[data-dash-column="ID"]').textContent)
			let contentNode = row.querySelector('td[data-dash-column="Content"]')
	
			video.onseeked = function () {
				img = new Image();
				//Create the thumbnail
				if (video.currentTime in cache) {
					img.src = cache[video.currentTime]
				}
				else {
					ctx.drawImage(video, 0, 0);
					img.src = canvas.toDataURL('image/jpeg');

					//add to cache
					cache[video.currentTime] = img.src
	
				}
	
				//Add thumbnail to content
				img.width = 128
				contentNode.children[0].innerHTML = '';
				contentNode.children[0].appendChild(img)
	
				video.onseeked = null
				resolve()
			}
	
			let time = (frameIdx * 0.04).toFixed(2) //So that similar floats are equivalent during comparison
			if (video.currentTime == time) {
				video.onseeked()
			}
			else {
				video.currentTime = time
			}
		}
		else{
			resolve()
		}	
	})
	
}

openFullTranscript = function (event) {
	event.preventDefault()
	document.getElementById('modal-body').innerHTML = event.currentTarget.dataset.t
	toggleModal()
}

var setIndicatorThumbnailsTimeout = null;
setIndicatorContent = async function () {
	if (setIndicatorThumbnailsTimeout == null) {
		setIndicatorThumbnailsTimeout = true
		let indicatorRows = document.querySelectorAll('#indicators tr')
		if (indicatorRows.length > 1) {
			document.querySelector('.lds-container').style.display = 'block'
			var indicatorsNode = document.getElementById('indicators');
			indicatorsNode.style.display = 'none'
			
			for (var i = 1; i < indicatorRows.length; i++) {
				await setThumbnail(indicatorRows[i])
			}

			setIndicatorThumbnailsTimeout = null
			indicatorsNode.style.display = 'block'
			document.querySelector('.lds-container').style.display = 'none'
			let currentPage = document.querySelector('.current-page')
			//Setup page changed listener (pagination)
			if (currentPage != null) {
				if (currentPage.changeMutator == null) {
					currentPage.changeMutator = new MutationObserver(function () {
						setTimeout(setIndicatorContent, 100)
					});
					currentPage.changeMutator.observe(currentPage, { attributes: true });

				}

			}
		}
		else {

			indicatorsNode.style.display = 'block'
			document.querySelector('.lds-container').style.display = 'none'
			setIndicatorThumbnailsTimeout = null
		}
	}
}

var setTranscription = function (node) {
	node.dataset.t = node.textContent.repeat(1)
	let t = node.textContent.slice(0, 85)
	if (node.dataset.t.length > 85) {
		t = t + '...'
	}
	node.textContent = t

	setTimeout(function () {
		node.removeEventListener('click', openFullTranscript)
		node.addEventListener('click', openFullTranscript)
	}, 100)

}



var setTranscriptionTimeout = null;

var setTranscriptions = function () {
	if (setTranscriptionTimeout == null) {
		let transcriptions = document.querySelectorAll('#segmentation-table td.column-4 div')
		for (var i = 0; i < transcriptions.length; i++) {
			setTranscription(transcriptions[i])
		}

		//Don't repeatedly run segmentsChanged (Debouncer)
		setTranscriptionTimeout = setTimeout(function () { setTranscriptionTimeout = null }, 300)
	}
}

document.addEventListener('DOMContentLoaded', function (event) {
	//Load text lines
	var request = new XMLHttpRequest();
	request.open("GET", '/static/sentences.csv', false);
	request.send(null)
	var csvData = {}
	var jsonObject = request.responseText.split(/\r?\n|\r/);
	for (var i = 1; i < jsonObject.length; i++) {
		let data = jsonObject[i].split(',')
		csvData[data[2]] = data[1]
	}

	//Initialisation
	setTimeout(function () {
		document.querySelector('.lds-container').style.display = 'none'
		const indicatorsNode = document.getElementById('indicators');
		let v = document.getElementById('media');
		v.onpause = function () {
			let s = v.src.split('#');
			if (s.length > 1) {
				let end = parseInt(s[1].split(',')[1])
				if (parseInt(v.currentTime) == end) { //Close enough to the end (and only just past it)
					v.load()
				}
			}
		}

		v.onloadeddata = function () {
			setIndicatorContent()
			v.onloadeddata = null
		}
		v.load()

		//Setup mutation observer for shortening Transcripts of segments
		const targetNode = document.getElementById('segments');

		const segmentsChanged = function (mutationsList, observer) {
			setTranscriptions()

		}

		const segmentChangedObserver = new MutationObserver(segmentsChanged);
		segmentChangedObserver.observe(targetNode, { childList: true, subtree: true });


		//Setup mutation observer for displaying thumbnails of indicators
		const indicatorsChanged = function (mutationsList, observer) {
			v.onloadeddata = function () {
				setIndicatorContent()
				v.onloadeddata = null
			}

		}

		const indicatorsChangedObserver = new MutationObserver(indicatorsChanged);
		indicatorsChangedObserver.observe(indicatorsNode, { childList: true, subtree: true });


		setTranscriptions()

	}, 400)



})
