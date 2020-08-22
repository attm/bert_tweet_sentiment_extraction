from pip._internal.utils.misc import get_installed_distributions
print_log = ''
for module in sorted(get_installed_distributions(), key=lambda x: x.key): 
    print_log +=  module.key + '~=' + module.version  + '\n'
print(print_log)