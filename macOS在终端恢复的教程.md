## 找到系统安装文件的url
1. 终端执行命令 拦截下载请求
```
tail -f /var/log/install.log | grep .pkg
```
2. 进入App Store 下载macOS系统, 
![](img/2022-06-08-13-00-39.png)
复制拦截到的url
macOS12
https://swcdn.apple.com/content/downloads/16/08/012-06873-A_636SHHRD4L/528ojpmw00mulgfjsz9k50modkj31a9v0p/InstallAssistant.pkg
macOS13
https://swcdn.apple.com/content/downloads/38/13/012-20267-A_8II0GZVCTD/vsifpgvw3a3xjgyznf6415vin9xv7a6ws5/InstallAssistant.pkg

## 进入恢复模式后
在终端中输入 resetpassword，回车
点击重置密码窗口，然后选择恢复助手—抹掉 Mac
在打开的窗口中点击抹掉 Mac，然后再次确认。完成后，Mac 会自动重启
之后输入命令
```
cd '/Volumes/Untitled'
mkdir -p private/tmp
cp -R '/Install macOS.app' private/tmp
cd 'private/tmp/Install macOS.app'
mkdir Contents/SharedSupport
curl -L -o Contents/SharedSupport/SharedSupport.dmg + 上面的连接
```

这时，Mac 会开始下载  macOS，完成后，输入下方命令：
```
./Contents/MacOS/InstallAssistant_springboard
```
当 macOS安装器打开时，可以跟随指令重新安装 macOS
