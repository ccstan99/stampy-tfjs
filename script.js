/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

const TITLE = document.getElementById('title');
const STATUS = document.getElementById('status');
const QUESTION_TEXT = document.getElementById('question');
const RESULTS = document.getElementById('results');
const ANCHORS = RESULTS.getElementsByTagName('a');
const ENCODE_BUTTON = document.getElementById('encode');
const SEARCH_BUTTON = document.getElementById('search');
const DOWNLOAD_BUTTON = document.getElementById('download');

ENCODE_BUTTON.addEventListener('click', encodeQuestions);
SEARCH_BUTTON.addEventListener('click', semanticSearch);
QUESTION_TEXT.addEventListener('keyup', semanticSearch);
QUESTION_TEXT.addEventListener('blur', () => {RESULTS.classList.remove("show")});
DOWNLOAD_BUTTON.addEventListener('click', downloadEncodings);

/* ============================================================================= */
// Store the resulting model & question embeddings in the global scope
var model = undefined;
var allEncodings = undefined;
var allQuestions = []

// TODO: should download from wiki, encountering CORS policy err
// TODO: consider embed answers along with questions
//QUESTIONS_URL = "https://stampy.ai/w/api.php?action=ask&query=[[Canonical%20questions]]|format%3Dplainlist|%3FCanonicalQuestions&format=json"
//QUESTIONS_URL = "https://stampy.ai/w/api.php?action=ask&query=[[Canonically%20answered%20questions]]|format%3Dplainlist|%3FCanonicalQuestions&format=json"
//QUESTIONS_URL = "https://stampy.ai/w/api.php?action=ask&query=[[Category:Answers]][[Canonical::true]][[OutOfScope::false]]%7Climit%3D1000%7Cformat%3Dplainlist%7C%3FAnswer%7C%3FAnswerTo&format=json"
//QUESTIONS_URL = "https://www.cheng2.com/share/stampy-answered-canonical-qas.txt";
let QUESTIONS_URL = "https://www.cheng2.com/share/stampy-canonical-qs.txt";
let ENCODINGS_URL = "https://www.cheng2.com/share/stampy-questions-encodings.txt";

/* ============================================================================= */
// create sentence embeddings for all questions in stampy db
async function encodeQuestions()
{
  STATUS.innerText = "Encoding Stampy DB questions..."

  // allQuestions must already downloaded from wiki or other source
  let f = await fetch(QUESTIONS_URL);
  let text = await f.text();
  
  console.log("load questions");
  response = JSON.parse(text)["query"]["results"]["Canonical questions"]["printouts"]["CanonicalQuestions"];
  
  // reinitialize encodings & question lists
  allEncodings = undefined;
  allQuestions = []
  for (let i=0; i < response.length; i++)
    allQuestions.push(response[i]["fulltext"]);
  console.log(allQuestions);

  enableSearch(false);

  // encoding questions in lowercase
  let formattedQuestions = []
  for (let i=0; i < allQuestions.length; i++)
    formattedQuestions.push(allQuestions[i].toLowerCase());

  // may run out of memory, set batch=true to encode sentences in mini_batches
  let batch = false;  
  if (batch)
  {
    let batch_size = 50;
    let i, encodings;

    // 2D tensor consisting of the 512-dimensional embeddings for each sentence.
    for (i=0; i+batch_size<allQuestions.length; i+=batch_size)
    {
      console.log("encoding "+i+".."+(i+batch_size))
      encodings = await model.embed(formattedQuestions.slice(i,i+batch_size))

      // init if null else concat results
      allEncodings = (!allEncodings) ? encodings :  allEncodings.concat(encodings);
    }
    // catch leftover batch to end
    console.log("encoding "+i+"..end")  
    encodings = await model.embed(formattedQuestions.slice(i))
    allEncodings = allEncodings.concat(encodings)    
  }
  else
  {
    // emcoding all questions in 1 shot without batching
    console.log("encoding 0..end")  
    allEncodings = await model.embed(formattedQuestions)
  }

  allEncodings.print(true /* verbose */);

  enableSearch(true);
}

/* ============================================================================= */
// load & save questions along with embeddings
async function loadEncodings()
{
  STATUS.innerText = "Encoding Stampy DB questions..."

  let f = await fetch(ENCODINGS_URL);
  let text = await f.text();

  console.log("load encodings")
  let response = JSON.parse(text);
  allQuestions = response["questions"];
  allEncodings = tf.tensor(response["encodings"]);
  allEncodings.print(true /* verbose */); 
  enableSearch(true);
}

/* ============================================================================= */
function enableSearch(allowed)
{
  QUESTION_TEXT.disabled = !allowed;
  SEARCH_BUTTON.disabled = !allowed;
  DOWNLOAD_BUTTON.disabled = !allowed;
  if (allowed)
    STATUS.innerText = "Ready to search.";
    // if not allow, calling func should set message
}

/* ============================================================================= */
async function semanticSearch()
{
    // can't search until all encodings for questions exist
  if (!allEncodings) return;
  
  console.log("semantic search")
  //loadStampy.goToAndPlay(1, true);
  
  encodeAndSearch();
}

async function encodeAndSearch()
{
  // STATUS.innerText = 'Searching...';
  // embed question, search for closest QnA match among embedded list
  let question = QUESTION_TEXT.value;
  
  // remove spaces and to lowercase
  question = question.toLowerCase().trim().replace(/\s+/g,' ');
  console.log("embed: " + question);
   
  // encodings is 2D tensor of 512-dims embeddings for each sentence
  let questionEncoding = await model.embed(question);
  questionEncoding.print(true /* verbose */);

  // tensorflow requires explicit memory management to avoid mem leaks
  tf.tidy(() => {

    // numerator of cosine similar is dot prod since vectors normalized
    let scores = tf.matMul(questionEncoding, allEncodings, false, true).dataSync();
    let scores_list = [];
    // wrapper with scores and index to track better
    for (let i=0; i<scores.length; i++)
      scores_list.push([i,scores[i]]);
    // sort by scores, ideally would like to use tf.nn.top_k
    let topScores = scores_list.sort((a,b) => {return b[1]-a[1]})

    let k = 5
    // print top k results
    for (let i = 0; i < k; i++)
    {
       let resultString = "(" + topScores[i][1].toFixed(2) + ") " + allQuestions[topScores[i][0]];
       ANCHORS[i].innerText = resultString;
    }
    RESULTS.classList.add("show");
  });
  questionEncoding.dispose();
}

/* ============================================================================= */
function download(data, fileName)
{
    let downloadAnchorNode = document.createElement('a');
    downloadAnchorNode.setAttribute("href", data);
    downloadAnchorNode.setAttribute("download", fileName);
    document.body.appendChild(downloadAnchorNode); // required for firefox
    downloadAnchorNode.click();
    downloadAnchorNode.remove();  
}

/* ============================================================================= */
async function downloadEncodings()
{
  // load & save questions along with embeddings
  let exportObj = JSON.stringify({ questions: allQuestions, encodings: allEncodings.arraySync() });
  let fileName = "stampy-questions-encodings.json";
  let data = "data:text/json;charset=utf-8," + encodeURIComponent(exportObj);
  
  console.log(data);
  download(data, fileName);

  await exportEncodingsTSV();
  await exportQuestionsTSV();
}

/* ============================================================================= */
async function exportEncodingsTSV()
{
  // export embeddings as TSV for visualization https://projector.tensorflow.org/
  console.log("exportEncodingsTSV");
  let exportObj = allEncodings.arraySync();  
  let fileName = "stampy-encodings.tsv";
  let tsv = "";
  exportObj.forEach(function(row) {  
    tsv += row.join('\t');  
    tsv += "\n";  
  });  
  let data = 'data:text/tsv;charset=utf-8,' + encodeURI(tsv);  
    
  console.log(data);
  download(data, fileName);
}

/* ============================================================================= */
async function exportQuestionsTSV()
{
  console.log("exportQuestionsTSV");
  let exportObj = "Questions\n" + allQuestions.join('\n');  
  let fileName = "stampy-questions.tsv";
  let data = 'data:text/tsv;charset=utf-8,' + encodeURI(exportObj);  

  console.log(data);
  download(data, fileName);   
}

/* ============================================================================= */
// Wait for USE universal sentence encoder class to finish
// loading. Machine Learning models can be large and take a moment 
// to get everything needed to run.
// Note: use is an external object loaded from our index.html
// script tag import so ignore any warning in Glitch/CodePen.
use.load().then(function (loadedModel) 
{
  model = loadedModel;
  // Model is now ready to use.
  ENCODE_BUTTON.disabled = false;
  TITLE.innerText += ": tf.js " + tf.version.tfjs;
  STATUS.innerText = "USE model loaded.";
  loadEncodings();
});

/* ============================================================================= */
const STAMPY_BOUNCE = 0;
const STAMPY_BLINK = 1;
const STAMPY_STOP = 2;

const BOUNCE_BUTTON = document.getElementById('bounce');
const BLINK_BUTTON = document.getElementById('blink');
const STOP_BUTTON = document.getElementById('stop');
BOUNCE_BUTTON.addEventListener('click', () => {stampyAnim.jumpToInteraction(STAMPY_BOUNCE)});
BLINK_BUTTON.addEventListener('click', () => {stampyAnim.jumpToInteraction(STAMPY_BLINK)});
STOP_BUTTON.addEventListener('click', () => {stampyAnim.jumpToInteraction(STAMPY_STOP)});

var stampyAnim = LottieInteractivity.create({
  mode: 'chain',
  player: '#logo',
  actions: [
    {
      // bounce in
      state: 'autoplay',
      frames: [30, 60],
      transition: 'none',
      repeat: 1,
    },
    {
      // loop blink for search
      state: 'loop',
      frames: [60, 100],
      transition: 'none',
    },
    {
      // paused animation
      state: 'none',
      frames: [100, 100],
      transition: 'none',
      repeat: 1,
    },
  ],
});

