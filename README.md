### **Canonical Correlation Analysis in Image Annotation and Classification** 



**Shan Lu, Applied Mathematics, Class of 2015**

**School of Information, Renmin University of China**

### Step 1 Source Code
As usual, there are mainly two ways to get codes from GitHub:
* Issue the command `git clone https://github.com/baizhima/cca`
* Download the zip archive through the link on this project page

### Step 2 Datasets
The dataset that used to train and test this CCA model is collected by National University of Singapore called NUS-WIDE(http://lms.comp.nus.edu.sg/research/NUS-WIDE.htm). However, this way is not recommended as images are still in pixels.

At minimum, you should download files as follows:
* NUS-WIDE training set: [Annotations](http://www.mmc.ruc.edu.cn/tagrel/data/flickr81train.Annotations.tar.gz), [MetaData](http://www.mmc.ruc.edu.cn/tagrel/data/flickr81train.MetaData.tar.gz), [DSIFT](http://www.mmc.ruc.edu.cn/tagrel/data/flickr81train.dsift.tar.gz)
* NUS-WIDE testing set:  [Annotations](http://www.mmc.ruc.edu.cn/tagrel/data/flickr81test.Annotations.tar.gz), [MetaData](http://www.mmc.ruc.edu.cn/tagrel/data/flickr81test.MetaData.tar.gz), [DSIFT](http://www.mmc.ruc.edu.cn/tagrel/data/flickr81test.dsift.tar.gz)

Basic useful feature list:

 * Ctrl/Cmd + S to save the file
 * Drag and drop a file into here to load it
 * File contents are saved in the URL so you can share files

中文

I'm no good at writing sample / filler text, so go write something yourself.

Look, a list!

 * foo
 * bar
 * baz

And here's some code! :+1:

```javascript
$(function(){
  $('div').html('I am a div.');
});
```

This is [on GitHub](https://github.com/jbt/markdown-editor) so let me know if I've b0rked it somewhere.


Props to Mr. Doob and his [code editor](http://mrdoob.com/projects/code-editor/), from which
the inspiration to this, and some handy implementation hints, came.

### Stuff used to make this:

 * [markdown-it](https://github.com/markdown-it/markdown-it) for Markdown parsing
 * [CodeMirror](http://codemirror.net/) for the awesome syntax-highlighted editor
 * [highlight.js](http://softwaremaniacs.org/soft/highlight/en/) for syntax highlighting in output code blocks
 * [js-deflate](https://github.com/dankogai/js-deflate) for gzipping of data to make it fit in URLs
