train_data = {
'����!' : '� ����, �������, ���������� � ������ �������, �� ��� ����� �� ��������� ��������'
'��� �� ���� �������?��� �� ������ �������� �� ���� ������� ����� �� ���������� �����!'
'��� �� �� ������� ���� �����? 	����� ��� ����� ��������� ����, ������. ��� ������ � ������ ����� - � ����, � ��������'
'����� �� ����� ������ ����?	�� ���� ���������� ����� ����� ����������. ������ ����� ������� ������� �����������. � ����� ����� ���� ��� ��������� ��-������'
'��� �� ���� ����� � ������?	����� �� �������, ��� ���� ������ �� ����� ���������� ����� - ������, ������ ��� ������. ���� ������ ������'
'��� ������� �� ���� ���������?	���� ����������� �������, � ��������� ���������� � ��������� ���������. ���� � ������� ��������, � ������, ������ ��� ���� � ���� - ������ �����������'
'��� ������� �� ���� �����?	� ����� ������ ��� ����� - �� ����� � �� �������. ���� ����� ������������ ������� ������ - ���, ��� � ��� �����������, �� ������� - ������� �����! ���� ����� ���������, ���� ����� ���������� � ������� �� ����� ��������� - ��������� ��������� � ��� ������'
'��� �� �� ������� ������?	������ ������� ������ ��������� - ������ ����� ���� ���������, ������ ����� �������� ������, ������ �� ���������� ��������. �� ������, ������� ����� ������ ������� ����, �� ��� � 100 ���, �� ����'
'��� ������� � ����������?	������ � �������� ������������ ����� ����������� ������, ��� �� �����������. ��� � ��������� �������� ������ � ����������. ������������ ����������� � ������ ������, ��� ������ ���� ������. � ����� ���� ������������. ����. ���������. ��� ��������� ���������� - �������� ��������. ���������. ��� ��� ����� ������? � ������� �����? ��� �� �������?'
'��� ����� ������ � ����� ������?	������ �����, ��� �������������, �����a�������������� ����, ��� ��������� �������������, ����������, ������, ��� �������, � ��� �������? � �������� ���, ������� ������ � ����'
'��� �� �� ���������� �� ��� ����?	� ����� � ������������� ������������� ����������� ��������� ��������� ������������ �������� ������ ����� ������������ �������. ���� ���������-����������� - ������ ���������, ������ - ��������, "������ ������" - ����������, � ���� - ����������. �� � ��� �������� �������-������ - ��� ��� ������ ��������� ������ ��� ��������, �� �� ��� ��� ��� ������� �� ���������'
'��� ������� �� ���� ������?	��� �������� ����������� (����������� - ����. "��"), ����� � �����������. ������ ���� ���������������� �����. ����� ����, ����� ������? ���� ���� ������. ������� ��� ������ ��� ����� ����������� ������������ ���� �������� � ������������ �� �������. ����� ������ � ��������� � ���� ������? ������ ������ �������� � �������� � �����'
'����� ������ �� ������ ����� ����������?	������� � ������� ������ �� ���� �������� ������ �������, ����� ���������, ������� ������� ����� ����� ����� �� ����� ������ �������� - ����� ���� ������ �����. �� ������, � ����� ������� ������ ��������'
'��� �� ���������� ���� �����?	���, ��� � ����� � �����, �� ����������. ��� � ���� �� ������� ������, �� ������ �����, �� ������� ���������, �� �������, ������� �������� ��. ���, ��� � ����, - ��� �� ������. �� ������ � ������ �����'
'������ �� �� ����� �����������?	� ���� ������ ����, �� ��� ����� � �����, ���� � ����� �����������'
'���������� ��� ������?	��� ������ ���� � ���� � �����, ��, ������ � ������!'
'����� ��� ������� �����?	����� �������� ���� ���� ���, � ������� � ����� � ������ ����'
'��� ���������� � ��������?	������� ������ ������ ����, �������, ������� � ��������.'
'�� ����� ���������� � ��������	����� �� ���� ��������� ��������, ��� � ���-�� �� ��� � �������� ��������� � ��� ��� ��������� ��� ������� ����. ��� �� ��� ��� ����������. ����� ��� �������, ��� �����, ��� ���� ��� ������� ����. ������ ����. � ���� ��� ������ ������ �� ��� �������'
'��� ���������� � ������� ���������	����, ��� �� ����� ����� ���� ������� ����. ������ ��� ��� ���� �� ���� ���������'
'������ � ��� ������?	������ � ��� Maybach? � ������, ������ � �� ��������!'
'��� ������� �� ���� ����� �����	����� ���� ������ ����-���� ������� �������������� ���� �����, � ���� ������ [���] ����� ��� �����!'
'��� ���������� � �������?	�� � ����������� ������ ������� ���������� ���������� ������� ������ ������, �������, ���������� �� �����!'
'��� �� �� ������� ����� �������������?	������ � ��������, ���� ��� �� �������.'
'��� ���������� � ����?	��� ��������� ����� ���������� ���� �� ���������. �� ��� ����� ����������� ��������.'
'� ����� ������ ���������, ��� �� �� ��� �������?	� ������, ��������� ������� ������� � � ��� ���������, �������� � ������, � ���� ��������� ��������, � [��] ���������� �������. ������ ���� ����� ����, �� �� ������� ������!'
'��� �� ������ ������ �������?	������� �� ������ �������� ����.'
'��� �� �� ������� ����?	� �����, ����� � ����. � ���� ��� ��������.'
'� ������ � ����������	����������� �������� �� ����������������� � ����� ����������. �� ������, ����� �� ����� �������, � ������ �����, ����� � ���������. � ������ � �� �������! ������ 22% ������� ������ �������� � ������� � ������. ������� ������� ���, ����� �� �������. ������� � ���������� ������ � ����� ��� ���� �� 50 ����� ������. � �������! � ������� ������� ��������'
'� ������� �� ��������� �������	�� ���������� ������ ������ �� ��������� �������. ����� ����� ������� ������, ����, ���, ���. � ������ � ��� ���� � �����������. ������������� ������! � ���� �� ���� ������������� �� ����� �� ������: ��� �������, ��� ���� �� ����. ��� ������. ��� ��� �������� � �����������'
'� �������� �� ����	��, ������� ����, ��������� �� ��, ����� ��� �������� ����� ��������� � ������� ��������� ���������, �� ��� 450 ������� � ��� ���� ������� ��������� �� ������, ��� �������� �� ��������. ������ ����� ���� �� ��� ��������. �� �� 102 ���� ������� ������� ���������, ���-�� ��������, ���-�� ������蠗 � ������ �������� � ����� ���. ����� ��� ��������� ������, ����� ��� ������ �����������'
'� �����������	����������� ���������� ��� ����� ����� �����: ����� ���������������� ����������, ���������������, �������������� �����, �������� ��������, ��� �������� ����� � �������. ����������, ��� ������������ ������� ������'
'��� ������ ���������� ����	���� ��������� ���� ������ ��������� �����������: �������� � ������. ���� ������������ �����. �� �� ����� ����� ������ �������� ����. �������� ���� ��� � ���������� ���������. ��������� ��-������ � �����������. ��� ����� �������� �������� � ���������� ���������. ���������� ��������, ����������, ���������, �������� � �.�. ����������� � ������ ���������� �� �����'
'� ������������� ������ � ������ � ������������	16������ ����� �������� �9 ���.  ��� 1945 ���� ����������� ���������� ����, �16������ 2014 ���� ������ ����� �������� ���������� ����� �������'
'��� ��������� ������� � ��������� ������	���� ������ ����������� �������������� �� ����� ������� ���������� ������������, �� ������� ������ ������� � �������� � ������ ������� � �������� �����. ����� ������� �������� ������� ��������, � ��� ��������� ����������� ��� ������� � �������, ��������� ������ � ��� �����. � ����� �� ��������� ��� ������� �����������, �����, ������ � ��� �����. ����� ������� ����� ����� ������� �� ����� ����'
'��� ����������� ���������� ����������������	������ � ������ ��� ���� �������������� ����������? ����� 0,2%, ����� ��� � ������ ������� ���� ���������� ��������� 20%! ��, ������ � ��� ����������� ������, � ��� ���� ������, ��� � ���, �� � ������� ������ �� ���� ��� ������.������ �� ��������� ��������� ����� ������? ������ � ��� � ������� ����� ��������, ���� � �������? ��������� �� �������� ����� ������ �����, �����������, ��������������, ������� ���������� � �����, ��������� ��������� ���������, ������� �� ������� � �������� �������. ����� ������� ������ ������ ��� ����������� �������: �������� � ��������, �����'
'��� ���������� ���������� � �������������� ������	�� ������ � ���� ��� �� ��� ����, ����� ��������� ���� � ����. ������ ���� ��������'
'��� ���������� ���������� � �������������� ������	�� ���, ��� ������� ������ ��������� ������ ��������, � ����� ���������� ����� ��� �����. ����������� ������� ������ �� ������ �������� �������� ����� �������, �� � ������� ������������� �� �����. ����� ���� ����������� ��������������� � ��������� ����������� ���������������, � �����������-������������ ���� ������ ������'
'���������� ��������	������� ����! ���������� ����� ���������� ����. ��� �������� ��������, ����, ������ ����� � ������� �����. ���� �� �������� ���� ���� ������ ����� ����� ��������� ��� ������ ��������������, �� ������ ��������� �����. � ������� � ������ � 17 ��� � ������ ���������, ��� ����� �� �������, � ��� ���� ���, ��� � ����. � ��� ����������, �� ���� ��������� ��������, ���� ��� ����� ��������, ���� ������ � ����� ������ �� �������, ��� ��� ��� � ������ ��� ������, �������� �� �����. ����� ����, ������ �������, ������������, ������ �� ������� � �� �����������'
'��� �� ���� ����� � ������?	����� �� �������, ��� ���� ������ �� ����� ���������� ����� � ������, ������ ��� ������. ���� ������ ������'
'��� �� �� ������� ������?	������ ������� ������ ��������� � ������ ����� ���� ���������, ������ ����� �������� ������, ������ �� ���������� ��������. �� ������, ������� ����� ������ ������� ����, �� ��� � 100 ���, �� ����'
'��� �� �� ������� ������?	������ ������� ������ ��������� � ������ ����� ���� ���������, ������ ����� �������� ������, ������ �� ���������� ��������. �� ������, ������� ����� ������ ������� ����, �� ��� � 100 ���, �� ����'
'� ��� ������� �������	� ���� ���� ������� ��������, �� ����� "���" � �����������. �� ���� �� ����������. � ���� ��� ������ � ���� �������� ��������.'
'��� �� ���������� � ������� �����?	� ������������ ������ ���, ��� �����. ��������� �������� �������������, � � ��� ���� ������. ���� ���� ����� � ������ ������, ����� ���������, ��� � ���.'
'������ �� �� ������������ � �������?	� �� ��������� (�������. � ����. ���.), � �� ����� ������ ���� ����������� � �������, ���� � �������� ���� �����������, ��� � ������ ���� ������ ��� �������, ��� � ��� ����.'
'��� �� �� ������ ������ ������ ����?	"� ���� ����������, ����� ���� ���������� �� ������� �������, ������ �� ��������. ������� ����� ����� �� ��������, � � ������ � ��� ������� ����� ������ ��������, ��� �������'
'��� ���������� � �����?	�� ���������� � ������������ �������, � ���� �������, � ���. �� ���� ������ � ���� �������� � ������'
'��� ���������� � �����?	� � ����� ���� ������ ��������� ��� ������� �����: ��� � �������. ��� �������� �����������, ����������� � ������ ����� ������� ���������� � �������� ��������� �������������� �����: ��� ������, ��� �����. ��� ����, ��� �����.'
'� ��� ��� ����� ��� ������	������ ��� ����� ��� ������? "���������� ������, � ������ ������� ������� ���". �? ������ ��� ��� �����. ��� ������� ��� ���� ��� �����. ����������. ��������. ����� ���, � �������� ����'
'��� ������ ������ � ��������?	� ������� ������� ���������. ������ ����-���� � �������, ����� ��� ���������� 90 ���. ��� �����������, ����� ������ �������������. ������ ��� ������. �� ���� ������ �� �����'
'� ��� ���������� ������� ������� ��������, �� ��� ��?	���� ������� �������� ������ ������ ������, ��� ������ ���������� ������ ������� �������. ������� ������ ���������� ���� � ������������ ���������� ��� ����� ������ �������.'
'��� ���������� � ���������� �������?	��������� � ��� �����, ���������� � �����'
'��� ��� � ���?	������� ��������� ���� � ��� ����� ����� �������. � �� ���� ����� ������ � ������� ����. ����� � ���� ���-�� � ����� ���� ������������, �� �������� �����'
'��� ������� �� ���� ������ �����������?	������� ����������� � ����� ������ � ����! ��� �� ���������� ���� ����� ������-�� ���������� ����, 200 �����, ���� ������� ���� ������, ���� ������ ������. ��� �� ��� ������� ������ � ����� �� ���� ������� ������?'
'���� �� ������� ��� ������?	� ������, ����� ������� ������� ����� ���� ���� ������ ����� ���������� ������ � �������� ������� �� ������ ����� ������'
'���� �� ����� ������ �����?	�� ���� ���������� ����� ����� ����������. ������, ����� ������� ������� �����������. � ����� ����� ���� ��� ��������� ��-������'
'��� �� �� ������� ������?	������ ������� ������ ��������� � ������ ����� ���� ���������, ������ ����� �������� ������, ������ �� ���������� ��������. �� � ������, ������� ����� ������ ������� ����, �� ��� � 100 ���, �� ����'
'��� ���������� � �����?	� ����� ������ ��� �����: �� ����� � �� �������. ���� ����� ������������ ������� ������, ���, ��� � ��� �����������, � �� �������. ������� �����. ���� ����� ���������, ���� ����� ����������, � ������� �� ����� ���������. ��������� �����������, � ��� ������'
'��� �� ���������� � ����������?	��� ��������� ���������� � �������� ��������. ���������. ��� ��� ����� ������? � ������� �����? ��� �� �������?'
'��� ���������� � ����������	� �������� ���������, �� ���������, �� ���������� � ����� ������������� ������ � �������� ��������� ������, ������� ��������� ������������ � �������� �������� ������'
'��� ������� �� ���� �����4�� � ��������?	�� ������� ��������� ����� ������� ������� ���������. �� ����� �� ������� ����� ������. ���������� ��������: �����, ������� ��� �����-������ ���������. � ������� ����� ������ ����� �������� �������� ������'
'��� ���������� � ���������?	��� �������, ������������� ������� ������ ����, ���� ���� ������������ ����� ������� ������ �� �����'
'��� ���� ������� � ����������?	����� �� ����� ��������� ����������������� ������. �������� �� ���������� ��� �������. ��� �� �����? ��, ���, ���������� �����? �������� � ������������ ���������������� �����. � ��� �� ���� ������. �������. �� 100 �������. ��� ��� ������? ��, ����� ����������. ��� ��� ����� ���� �� ������ ������� � �����? ��� � ������ �� ������. ������� ���������� ��������� ����� �������� ����������� ������ � �����. ��������� ������'
'��� ������� � ��������?	���������� ������� ������� � ������ ������ ��������� ������. ����� ������ �� ������. ����� ������ �� ������'
'��� ������� � ��������?	��� ������������ � ���� ����������� ��-�� ������, ���!'
'��� ������� � ��������?	������� �� ���� ������� �� ������ �������, �������! ���� � �������, ���� �� ������, ���� � ��������'
'��� ������� � �������?	������ �������� ������� � �� �������. ������� ������� � �� ������� ������� �����'
'��� ������� � �������?	�� ����, ��� ������ �����, ���� ����� ����� 400 ������'
'��� ������� � ��������?	���� ���������� ������ ������� �������� �����, ������������ ���, �� ��������� ����������� �������, ��������� ��� ��� �����������, � ����� � ��������� ����� ��� ������� �� ������, ����� �� ���������� �����, �� ������� ���� �������� �������, ����� ������ � ������� �� �������'
'��� ���������� � �������?	��������� � ��� ����������. �������� � ��� �����, �� ��� ���� �� ��������� ������ �������. ��� ��������, ��� ������, ������. ������ ������ ����� ������� ������ �� ��������'
'����� �� ���� ����?	���� �� ����� �������, �� ����� ��� ����, ����� ������ ���� ��� ����, � ������� ����� � ����, � ������ � ����������'
'��� ������� �� ���� �������?	������� ���� �� ������. ��� ��� �������� �������� � ������ ������. � ��� ���� ������� � �� ��������, �� �������, �� ����������, ����� ������� ����� ����� �����������, ������� ����'
'����� �� ������� � ����?	������� ����� �� ����� ����. � ������� ���� ���, ��� ��������� �����������, ������ ��� ��� ���������� ��� ����� � ������� �����'
'��� ������� �� ���� �����?	�� ���� ������ � ������. ����� ������ ������. ��� �� ��������, ��� �� ����������. ������, ����� ������ ����� ������ �� �����. 250 ����� �������� ������ �����! ��� ��� ��������. ��� ��� ������� ������� �� ���� ���! ��� ������� ��� ���� �������, ���� ����� �����������, ����������. ������, �� ������. �� ����������, �..., �� ������, �� ������� ������ �������� �� ����� � ������ ��� ������ ����'
'��� ������� � ������� � ������� � 24 ����?	������� 2024 ���� � ��� �� �����, ������ ��� ����� ������� �� �����'
'��������� �� ������ ��� ����� ���?	������'
'�� ����� ����������� ����� ������ ������� ������ ��� �������������� ��� ����� ������� ��� ��� �� ���������� �� �������� ����� ��� ��� �� � ������������ �� �������?	��� ��� ������� ��� ������ �� ���� �� ���� ����� ������ �������� ������ �� ���'
'������ �� ������?	��� ���� ���� ��� �� ������ ������ ��� � ��� ��� ������������� ���� �� ��� ������ ���� ���'
'� ����� ��������� ��� ���� ��� �� ���� �� ��������� ��������� ����� ���������� ���� � ���� ����� � ������ ���������� ��������� ��� ������ � ����� � ������ ��� �������� ��� ��� ���������?	�� ������ ��� ����-�� ������� ��� ������ � ����� ����� �� ������ ������ ������� ���� ��� ���� ���� ���-�� ����� �������������� ����� ���� 6 ���� ������ �������� ������������ ������ ����� �� ������ ������� � ���� ������� ��������� ���������'
'� �� ����� ��������? 	�� ��, �� �� ���������� �����������'
'��� ��� ������ ������?	��� ���������� ����������� ������, � ��� ��� ���� ������� ����� ����� ������, � �������� �������� ������ ������'
'��� �� ������ ��� ����������? 	� �������� �� ������'
'���� �� ��� �� ������� ����?	����, ��� � ��� ����, ����'
'� ���� �� ����, �� ��� ���������?	�����, ��� ������ ���� ��������'
'��� ��������� ����� � ��� ���� �������� 95 ���� ��������?	�� ����� ������ �� ����� ������ ������� � 21 ���� ��� �������� ����� ����� ���� ���������� ��������� ��������������� ���� ��������� ������ ������ ��� 18 ��� ����� ����� ����� � � ��� �� ������ �� ����� ����� ��� ����� �������� ��� �������� � ��� 33 �������� �������� �� 15 �� 29 ��� ������ ���� ���������� �� ����� ����� ��� � ���� � ������� ��� ��� instagram ��� ������ ���� �� ��� � ����� ������ ��� ����� �� �� ������ ��� ��� ���� ������� �� ����� ��������� ���� ������� ����� � ������� ���������� �� ��� ������� � ������ � ���������� ��������� �� �� ����� �� ������� ����� ������� �������� ��������� ����� �������� db ����� ���� 0 �� ����� �������� ����� ������� ���� ������������ ������'
'��� ��� �����, ����� ���������� �� ����?	�� �� ������ ����������� �� ��������, �������� ������� - ����� �������� � ��������� �����������'
'����� �� ��� ����� ��������� ��������� �����?	���'
'���� � ��� ��� ��������, �� �������� � �����?	����� ���� � ������'
'�� ��� ���� �������� ������ ����������, ��� ����� �� ���� �������	�� � ���������� ���� �� ���� �������, ����������� '
'����� ��� �����?	�����'
'�������� ��� �������, ����� �� ���� ������ ������	���������� ��������, ������� ���� ���� ����������� �������� � 14 ���� � ����� ������'
'��� �� ���������� �� ��, ��� ��� �������� ����������� ������?	�� ��� �����, �� �� ��� ������� �� ������ � ������� �� ������� ��� ���-�� �������'
'�� ��� �����������, ��� ��� ������ ����������?	� ��� � �� ����������, � ����� ������ �� ��������'
'��� ����� �����������?	�������� ������'
'����� ������ �������� ������� �������?	��� ���� ������� ��� 20 ����� ��� ����, ����������� �� ���� ����, ���� ������ ������� ������� '
'� �� ����� ������ ��� ����������?	�� ���������� ������ � ��� ��������, ����������'
'������� ���� �� ����� ����?	��� �������� 4 ���������'
'������ ����� �� ��� ��� �� ������? 17 ���, ���� ������, ��� ���?	���� ������ ��������� ������, ��� ��������� ���, ���� ������ �����-������ ����� ������ ��� �� ����� ������� ��� ���������� �� ����� ����� �������� ������ ����� ��� ��� ������� ��� �� ����� ���� �� ����� �������'
'� ����� ��� ����� ����������?	� ������? ����, ������, �����, ������ ��� ��� ����, ����� ����� ������� ��� �����'
'�� ��� ����� �� �� �� ������� ����, ��� �����? ��������� �� �����	���, ���! ��� ���� �� �� �� ����� ����, ������ �������� ������� ���� �� ������, �����, � ��� ������� ������ �� ����, � ��� ����� �� ��������� ������� ����'
'� �� �������� �������� ������ ���� ������ �������������� ������� ��������?	���� - ��� ���������� � ������ ����, ���� � ��� ���� ����, �� �� ����� ���� ����������!'
'� 2017 ���� � ���� ���� ������� ������, �� ��������, ��� ����� ������� ������?	� ��� �����, � ��� �����������.  �� ������� ��� ������� ����� ������� � ����� �����������. ���! ����� ����� ������� ���������� ������, ������� ������ ����� �� ��������� �����. ����� ��� ������� ������, ����� �� ��� ���������� �������!'

}
test_data = {}