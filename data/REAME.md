# Data processing

`get_evaluation.sh`:

`get_embeddings.sh`: Downloads "Multilingual word Embeddings" files and build a binary equivalent. It lower size of the raw models by ~10

It relies on the npm library `protobufjs` and uses the following proto definition:

```
syntax = "proto3";
package MuseModels;
option optimize_for = LITE_RUNTIME;

message Vector {
  required string word = 1;
  repeated float vector = 2;
}

message Simple {
  repeated Vector data = 1;
}
```

Reading the binary files is as easy as the following (Node script):

```js
const protoPath = path.join(
  __dirname,
  "./multilingual-word-embeddings.proto"
);

const binaryModelPath = path.join(
  __dirname,
  "./models/wiki.multi.en.min.vec"
);

const root = ProtoBufJs.loadSync(protoPath);
const protoModel = root.lookupType("MuseModels.Simple");

const buffer = fs.readFileSync(binaryModelPath);
const messageModel = protoModel.decode(buffer);
const model = protoModel.toObject(messageModel);
```