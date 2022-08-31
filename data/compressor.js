// To be run on demand. Language by language
// ex: run node build_binary_models.js en

// READ EMBEDDINGS
import fs from "fs";
import { createRequire } from "module";
import path from "path";
import ProtoBufJs from "protobufjs";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const require = createRequire(import.meta.url);

const absPath = path.join(
  __dirname,
  "./multilingual-word-embeddings.proto"
);
const root = ProtoBufJs.loadSync(absPath);
const protoModel = root.lookupType("MuseModels.Simple");

process.argv.shift();
process.argv.shift();
const enFilePath = path.join(
  __dirname,
  `./raw/wiki.multi.${process.argv[0]}.vec`
);

const VECTOR_LEN = 300;
let NMAX = 50000;

const LineByLineReader = require("line-by-line"),
  lr = new LineByLineReader(enFilePath);

lr.on("error", function (err) {
  console.log(err);
});

let allData = [];
let i = 0;

const trim = (s) => s.replace(/^\s+|\s+$/g, "");
// TODO: remove latin words in Arabic embedding !!!
const isWordValid = (word) => word.indexOf("/") < 0 && word.length >= 3;
const downcast = (num) => Number(num);

lr.on("line", function (line) {
  lr.pause();
  let lineArray = trim(line).split(" ");
  let word = lineArray[0];
  if (isWordValid(word)) {
    const vector = lineArray.slice(1).map(downcast)
    if (vector.length === VECTOR_LEN) {
      // Now having our (word, vector)
      allData.push({ word, vector });
    }
  }
  i++;
  if (i === NMAX) {
    lr.close();
  }
  lr.resume();
});

lr.on("end", function () {
  console.log("source finished");
  var err = protoModel.verify({data: allData});
  if (err) throw Error(err)
  const tmp = protoModel.create({data: allData});
  const buffer = protoModel.encode(tmp).finish();
  fs.writeFileSync(
    path.join(__dirname, `./models/wiki.multi.${process.argv[0]}.min.vec`),
    buffer
  );
});
