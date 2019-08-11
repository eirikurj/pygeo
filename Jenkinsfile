pipeline {
    agent any

    stages {   
        
        stage('py2_real_test') {
            steps {
                sh '. /home/mdolab/packages/setMdolabPath; conda activate py2; env|sort; cd reg_tests; python run_reg_tests.py -nodiff'
            }
        }
    }
}