import sys, os, shutil, time, argparse, re, paramiko

parser = argparse.ArgumentParser(description='')
parser.add_argument('--target_cluster', type=str, default="helios", help='Target cluster name')
parser.add_argument('--user', type=str, default="wilomit", help='Username')
parser.add_argument('--implementation', type=str, default="pytorch", help='tensorflow or pytorch')

args = parser.parse_args()

if args.implementation == 'pytorch':
    folder_exec = '~/scratch/GAT_twopaths'
elif args.implementation == 'tensorflow':
    folder_exec = '~/scratch/Stronger-GCN-TF2'

if args.target_cluster == "helios":
    hostname = 'helios.calculquebec.ca'
    str_running, str_pending = 'Running', 'Idle'
elif args.target_cluster == "beluga" or args.target_cluster == "graham":
    hostname = '%s.computecanada.ca' % args.target_cluster
    str_running, str_pending = 'R', 'PD'
if args.user == "wilomit":
    username, password = "wilomit", "!9kgQrnxtm94CGX"

for folder in ['generated', 'submitted', 'operating', 'completed', 'exception']:
    try:
        os.mkdir(folder)
    except OSError:
        pass

JOB_LIST = []

def connect():
    client = paramiko.SSHClient(); client.load_system_host_keys(); client.set_missing_host_key_policy(paramiko.WarningPolicy)
    global hostname, username, password
    client.connect(hostname, port=22, username=username, password=password)
    return client

client = connect()

def submit(client, script_path, script_name):
    global args
    sftp = client.open_sftp(); sftp.put(script_path, script_name)
    _, stdout, _ = client.exec_command("dos2unix" + " " + script_name)
    time.sleep(2)
    if args.target_cluster == "helios":
        try:
            _, stdout, _ = client.exec_command("msub -A jvb-000-ag" + " " + script_name)
            jobid = str(int(stdout.readlines()[1]))
        except AttributeError:
            return ''
    elif args.target_cluster == "beluga" or args.target_cluster == "graham":
        try:
            _, stdout, _ = client.exec_command("sbatch" + " " + script_name)
            jobid = str(int(re.search(r'\d+', ' '.join(stdout.readlines())).group()))
        except AttributeError:
            return ''
    shutil.move(script_path, "submitted" + "/" + script_name.replace(".sh", "_%s.sh" % jobid))
    client.exec_command("rm" + " " + script_name)
    return jobid

def isin(filename, folder):
    filelist = os.listdir(folder); found = False
    for name in filelist:
        if name == filename:
            found = True; break
    return found

def inquire(client, identifier, jobid):
    _, stdout, _ = client.exec_command("cat %s/%s.txt" % (folder_exec, identifier))
    RUNNING_CMDOUT = stdout.readlines()
    if len(RUNNING_CMDOUT) == 0:
        return 'incomplete'
    else:
        firstline = RUNNING_CMDOUT[0]
        if len(firstline) > 2:
            _, stdout, _ = client.exec_command("rm" + " " + "%s/%s.txt" % (folder_exec, identifier))
            if args.target_cluster == 'beluga' or args.target_cluster == 'graham':
                client.exec_command("rm" + " " + "slurm-%s.out" % jobid)
            elif args.target_cluster == 'helios':
                client.exec_command("rm" + " " + "%s.out" % jobid)
                client.exec_command("rm" + " " + "%s.err" % jobid)
            return firstline
        else:
            return 'incomplete'

def refresh(client):
    global JOB_LIST, hostname, args
    print('\n' + 'collecting unfinished tasks')
    JOB_IDENTIFIERS = [entry['identifier'] for entry in JOB_LIST]
    for folder in ['generated', 'submitted', 'operating']:
        filelist = os.listdir(folder)
        for filename in filelist:
            if filename.find(args.target_cluster) == -1: continue
            if folder == "generated":
                identifier, job_id = str(int(re.search(r'\d+', filename).group())), 'unknown'
            else:
                try:
                    identifier, job_id = map(int, re.findall(r'\d+', filename))
                    identifier, job_id = str(identifier), str(job_id)
                except ValueError:
                    identifier, job_id = str(int(re.search(r'\d+', filename).group())), 'unknown'
            if identifier not in JOB_IDENTIFIERS:
                entry = {'filename': filename, 'status': folder, 'identifier': identifier, 'job_id': job_id}
                JOB_LIST.append(entry)
                print('identifier: %6s, status: %s, job_id: %s' % (entry['identifier'], entry['status'], entry['job_id']))
    for i in range(len(JOB_LIST)):
        if JOB_LIST[i]['status'] == "generated":
            try:
                job_id = submit(client, "generated" + "/" + JOB_LIST[i]['filename'], JOB_LIST[i]['filename'])
            except OSError:
                client = connect()
                job_id = submit(client, "generated" + "/" + JOB_LIST[i]['filename'], JOB_LIST[i]['filename'])
            if len(job_id) > 1:
                JOB_LIST[i]['job_id'] = job_id; JOB_LIST[i]['status'] = "submitted"
                JOB_LIST[i]['filename'] = JOB_LIST[i]['filename'].replace(".sh", "_%s.sh" % JOB_LIST[i]['job_id'])
                print("submitted script %6s as job %s" % (JOB_LIST[i]['identifier'], JOB_LIST[i]['job_id']))
            else:
                print("submission for script %s failed" % JOB_LIST[i]['identifier'])

    print('\n' + 'processing unfinished tasks')
    if args.target_cluster == "helios":
        command = "showq -u" + " " + username
    elif args.target_cluster == "beluga" or args.target_cluster == "graham":
        command = "squeue -u" + " " + username
    _, stdout, _ = client.exec_command(command)
    ALL_CMDOUT = stdout.readlines()
    if len(ALL_CMDOUT) == 0:
        print('no queue information from remote')
        return
    for i in range(len(JOB_LIST)):
        if JOB_LIST[i]['status'] != "generated" and JOB_LIST[i]['status'] != "exception":
            print('%s job with identifier %6s job_id %6s: ' % (JOB_LIST[i]['status'], JOB_LIST[i]['identifier'], JOB_LIST[i]['job_id']), end='')
        if JOB_LIST[i]['status'] == "submitted":
            found_running, found_pending = False, False
            for j in range(len(ALL_CMDOUT)):
                if ALL_CMDOUT[j].find(JOB_LIST[i]['job_id']) != -1:
                    if ALL_CMDOUT[j].find(str_pending) != -1:
                        found_pending = True; break
                    elif ALL_CMDOUT[j].find(str_running) != -1:
                        found_running = True; break
            if found_running:
                JOB_LIST[i]['status'] = "operating"
                shutil.move("submitted" + "/" + JOB_LIST[i]['filename'], "operating" + "/" + JOB_LIST[i]['filename'])
                print('now operating')
            elif found_pending:
                print('still pending')
            # elif isin(JOB_LIST[i]['filename'], "exception"):
            #     JOB_LIST[i]['status'] = "exception"
            #     try:
            #         shutil.move("submitted" + "/" + JOB_LIST[i]['filename'], "exception" + "/" + JOB_LIST[i]['filename'])
            #     except FileNotFoundError:
            #         pass
            #     print('now exception')
            else: # this handles a job with unknown job_id
                result = inquire(client, JOB_LIST[i]['identifier'], JOB_LIST[i]['job_id'])
                if result == 'incomplete':
                    if JOB_LIST[i]['job_id'] == 'unknown':
                        print('perhaps opertating, exception also possible!')
                    else:
                        JOB_LIST[i]['status'] = "exception"
                        try:
                            shutil.move("submitted" + "/" + JOB_LIST[i]['filename'], "exception" + "/" + JOB_LIST[i]['filename'])
                        except FileNotFoundError:
                            pass
                        print('now exception')
                else:
                    JOB_LIST[i]['status'] = "completed"
                    try:
                        os.remove("operating" + "/" + JOB_LIST[i]['filename'])
                    except FileNotFoundError:
                        pass
                    output_file = open("completed" + "/" + JOB_LIST[i]['identifier'] + ".txt", 'w'); output_file.write("%s" % result); output_file.close()
                    print('now completed')
        elif JOB_LIST[i]['status'] == "operating":
            if JOB_LIST[i]['job_id'] == 'unknown':
                result = inquire(client, JOB_LIST[i]['identifier'], JOB_LIST[i]['job_id'])
                if result == 'incomplete':
                    print('perhaps opertating, exception also possible!')
                else:
                    JOB_LIST[i]['status'] = "completed"
                    try:
                        os.remove("operating" + "/" + JOB_LIST[i]['filename'])
                    except FileNotFoundError:
                        pass
                    output_file = open("completed" + "/" + JOB_LIST[i]['identifier'] + ".txt", 'w'); output_file.write("%s" % result); output_file.close()
                    print('now completed')
            else:
                found_running, found_pending = False, False
                for j in range(len(ALL_CMDOUT)):
                    if ALL_CMDOUT[j].find(JOB_LIST[i]['job_id']) != -1:
                        if ALL_CMDOUT[j].find(str_running) != -1:
                            found_running = True; break
                        elif ALL_CMDOUT[j].find(str_pending) != -1:
                            found_running = True; break
                if found_running:
                    print('still opertating')
                elif isin(JOB_LIST[i]['filename'], "operating"):
                    result = inquire(client, JOB_LIST[i]['identifier'], JOB_LIST[i]['job_id'])
                    if result == 'incomplete':
                        JOB_LIST[i]['status'] = "exception"
                        try:
                            shutil.move("operating" + "/" + JOB_LIST[i]['filename'], "exception" + "/" + JOB_LIST[i]['filename'])
                            if args.target_cluster == 'beluga' or args.target_cluster == 'graham':
                                _, stdout, _ = client.exec_command("cat slurm-%s.out" % identifier)
                            elif args.target_cluster == 'helios':
                                _, stdout, _ = client.exec_command("cat %s.err" % identifier)
                            CMDOUT = stdout.readlines()
                            output_file = open("exception" + "/" + JOB_LIST[i]['identifier'] + ".txt", 'w')
                            for i in range(len(CMDOUT)):
                                output_file.write("%s" % CMDOUT[i])
                            output_file.close()
                        except FileNotFoundError:
                            pass
                        print('now exception')
                    else:
                        JOB_LIST[i]['status'] = "completed"
                        try:
                            os.remove("operating" + "/" + JOB_LIST[i]['filename'])
                        except FileNotFoundError:
                            pass
                        output_file = open("completed" + "/" + JOB_LIST[i]['identifier'] + ".txt", 'w'); output_file.write("%s" % result); output_file.close()
                        print('now completed')
                else:
                    JOB_LIST[i]['status'] = "exception"
                    print('neither operating nor found in the folder')
    JOB_LIST = [entry for entry in JOB_LIST if entry['status'] != 'completed' and entry['status'] != 'exception']

while True:
    refresh(client)
    time.sleep(30)
client.close()