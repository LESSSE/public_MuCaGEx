#!/usr/bin/python3
##LESSSE
##8 November 2018
##MuCaGEx
##____________
##Main Cycle
##____________

from config import exp_config
import report
from setproctitle import setproctitle
from load import *
from tqdm import tqdm
import resource
import datetime
import time
import gc

def bytes(size,precision=2,unit='B'):
    suffixes=['B','KB','MB','GB','TB']
    suffixIndex = suffixes.index(unit)
    while size > 1024 and suffixIndex < 4:
        suffixIndex += 1 #increment the index of the suffix
        size = size/1024.0 #apply the division
    return "%.*f%s"%(precision,size,suffixes[suffixIndex])

def main():
    #Config exp report
    report.config(exp_config)
    setproctitle(exp_config["name"])    

    #Loading our datasets
    report.out("______Loading_Datasets_Info__________".ljust(55,'_'))
    datasets=[]
    for d in exp_config["datasets"]:
        dataset, stats, excep = load_dataset(d)
        datasets += [dataset]
        report.dataset(dataset.id,stats,excep)

    #Loading our model
    report.out("______Loading_Session_and_Model__________".ljust(55,'_'))
    model, stats, excep = load_model(exp_config["model"])
    report.model(model.id,stats,excep)

    last_safepoint = time.process_time()

    #For each Epoch
    try:
        for epoch in tqdm(range(1,exp_config['epochs']+1),desc="epochs",unit=" epoch"):
         
            report.out("______Memory_Used__________".ljust(55,'_')+"\nRSS Max: "+str(bytes(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,unit='KB')))

            report.out("______CPU_Time__________".ljust(55,'_')+"\nCpu Time: "+str(datetime.timedelta(seconds=time.process_time())))
            if (time.process_time() > exp_config['cpu_time_limit'].total_seconds()):
                break
            
            if (time.process_time() - last_safepoint > datetime.timedelta(hours=12).total_seconds()):
                excep = model.save()
                report.save_model(model.iters,excep)
                last_safepoint = time.process_time()

            report.new_epoch(epoch)
            
            ds = []
            for d in datasets:
                if d.epochs < epoch:
                    ds.append(d)
            
            while ds:
                #Save
                if exp_config['freq']['saving?'](model.iters):
                    excep = model.save()
                    report.save_model(model.iters,excep)

                instance = {}
                names = {}
                for d in ds:
                    i = d.next_instance("train")
                    names[d.id] = i[0]
                    instance[d.id] = i[1]

                #Losses
                if exp_config['freq']['losses?'](model.iters):
                    losses, excep = model.loss(instance)
                    report.save_losses(losses,model.iters,excep) 

                #Sample            
                if exp_config['freq']['sampling?'](model.iters):
                    samples, excep = model.sample(instance)
                    report.save_samples(samples,model.iters,excep)
                    
                #Valid
                if exp_config['freq']['validating?'](model.iters):
                    validations, excep = model.validate(instance)
                    report.save_validations(validations,model.iters,excep)
                    
                #Test
                if exp_config['freq']['testing?'](model.iters):
                    testing, excep = model.test(instance)
                    report.save_tests(testing,model.iters,excep)
        
                #Train
                if exp_config['freq']['training?'](model.iters):
                    excep = model.train(instance)
                    report.save_train(names,model.iters,excep)

                model.add_step()
            
                #if exp_config['freq']['losses?'](model.iters):
                #    losses, excep = model.loss(instance)
                #    report.save_losses(losses,model.iters,excep) 
                
                #model.add_step()

                ds = []
                for d in datasets:
                    if d.epochs < epoch:
                        ds.append(d)
        

    except KeyboardInterrupt as ki:
        report.exception(ki)

    except ValueError as ve:
        report.exception(ve)

    finally:    
        report.out("______Finale__________".ljust(55,'_'))
        
        #Final Save
        excep = model.save()
        report.save_model(model.iters,excep)

        instance = {}
        names = {}
        for d in datasets:
            i = d.next_instance("train")
            names[d.id] = i[0]
            instance[d.id] = i[1]
        
        #Final loss
        losses, excep = model.loss(instance)
        report.save_losses(losses,model.iters,excep)

        #Final Sample
        for sample in tqdm(range(1,exp_config['final_samples']+1),desc="samples",unit=" sample"):
                instance = {}
                samples, excep = model.sample(instance)
                report.save_samples(samples,model.iters,excep)
                model.add_step()

if __name__ == "__main__":
    main()
