INSTANCEID=i-0217e93986bc4a388
IDENTITYFILE=/path/to/.ssh/aws-ec2.pem
SSH-USER=carnd
OPEN=open  # OSX
# OPEN=cygstart  # Linux
port=8888
protocol=http

INSTANCE=`aws ec2 describe-instances --output text --instance-ids $(INSTANCEID) --query "Reservations[0].Instances[0].PublicDnsName"`

# show infomation of the instance
describe:
	@aws ec2 describe-instances --output table --instance-ids $(INSTANCEID)

# start the instance
start:
	-aws ec2 start-instances --output text --instance-ids $(INSTANCEID)

# stop the instance
stop:
	-aws ec2 stop-instances --output text --instance-ids $(INSTANCEID)

# wait and print current status until the instance running
wait-start:
	@while :; do STATE=`aws ec2 describe-instances --output text --instance-ids $(INSTANCEID) --query "Reservations[0].Instances[0].State.Name"` ; test $$STATE = "running" && exit || echo $$STATE ; sleep 2s; done

#wait-start:
# aws ec2 wait instance-status-ok --instance-ids $(INSTANCEID)

# ssh connection
ssh: start wait-start
	# ssh -i $(IDENTITYFILE) $(SSH-USER)@`aws ec2 describe-instances --output text --instance-ids $(INSTANCEID) --query "Reservations[0].Instances[0].PublicDnsName"`
	ssh $(SSH-USER)@`aws ec2 describe-instances --output text --instance-ids $(INSTANCEID) --query "Reservations[0].Instances[0].PublicDnsName"`

# update .ssh/config for ssh, scp, etc.
update-ssh-config:
	DNSNAME=`aws ec2 describe-instances --output text --instance-ids $(INSTANCEID) --query "Reservations[0].Instances[0].PublicDnsName"` ; sed -i "s/Hostname ec2.*.compute.amazonaws.com\$$/Hostname $$DNSNAME/" $$HOME/.ssh/config

# show Public DNS name
name:
	@aws ec2 describe-instances --output text --instance-ids $(INSTANCEID) --query "Reservations[0].Instances[0].PublicDnsName"

# open web site in local browser
open:
	$(OPEN) $(protocol)://`aws ec2 describe-instances --output text --instance-ids $(INSTANCEID) --query "Reservations[0].Instances[0].PublicDnsName"`:$(port)

# upload model files to server
upload-model:
	rsync -arv --exclude=__pycache__ -e ssh ~/Projects/udacity/CarND-Behavioral-Cloning/model/ $(SSH-USER)@$(INSTANCE):/home/carnd/CarND-Behavioral-Cloning/model

# upload notebook to server
upload-notebook:
	scp ~/Projects/udacity/CarND-Behavioral-Cloning/CarND-Behavioral-Cloning.ipynb $(SSH-USER)@$(INSTANCE):/home/carnd/CarND-Behavioral-Cloning

# upload notebook to server
download-notebook:
	scp $(SSH-USER)@$(INSTANCE):/home/carnd/CarND-Behavioral-Cloning/CarND-Behavioral-Cloning.ipynb ~/Projects/udacity/CarND-Behavioral-Cloning/

# download computed models to local
download:
	rsync -arv -e ssh $(SSH-USER)@34.249.2.212:/home/carnd/CarND-Behavioral-Cloning/out/ ~/Projects/udacity/CarND-Behavioral-Cloning/out