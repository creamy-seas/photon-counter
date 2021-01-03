# Project setup #

1. Create `.dir-locals.el` listing location of standard libraries and header files.

```elisp
((nil . ((company-clang-arguments . ("-I/usr/include"
                                     "-I/Users/CCCP/photon_counting/c_lib/include")))))
```

2. Run `ggtags-create-tags` in an c/cpp project. or `gtags`

3. Install `cppunit` lib

```shell
sudo apt-get install libcppunit-dev
```
