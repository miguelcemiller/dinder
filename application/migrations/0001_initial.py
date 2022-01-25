# Generated by Django 4.0.1 on 2022-01-25 07:50

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion
import uuid


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name='Profile',
            fields=[
                ('first_name', models.CharField(blank=True, max_length=200, null=True)),
                ('last_name', models.CharField(blank=True, max_length=200, null=True)),
                ('user_image', models.ImageField(blank=True, null=True, upload_to='static/images/users/')),
                ('dog_name', models.CharField(blank=True, max_length=200, null=True)),
                ('dog_age', models.IntegerField(blank=True, max_length=10, null=True)),
                ('location', models.CharField(choices=[('Alaminos', 'Alaminos'), ('Bay', 'Bay'), ('Biñan', 'Biñan'), ('Cabuyao', 'Cabuyao'), ('Calamba', 'Calamba'), ('Calauan', 'Calauan'), ('Cavinti', 'Cavinti'), ('Famy', 'Famy'), ('Kalayaan', 'Kalayaan'), ('Liliw', 'Liliw'), ('Los Baños', 'Los Baños'), ('Luisiana', 'Luisiana'), ('Lumban', 'Lumban'), ('Mabitac', 'Mabitac'), ('Magdalena', 'Magdalena'), ('Majayjay', 'Majayjay'), ('Nagcarlan', 'Nagcarlan'), ('Paete', 'Paete'), ('Pagsanjan', 'Pagsanjan'), ('Pakil', 'Pakil'), ('Pangil', 'Pangil'), ('Pila', 'Pila'), ('Rizal', 'Rizal'), ('San Pablo', 'San Pablo'), ('San Pedro', 'San Pedro'), ('Santa Cruz', 'Santa Cruz'), ('Santa Maria', 'Santa Maria'), ('Santa Rosa', 'Santa Rosa'), ('Siniloan', 'Siniloan'), ('Victoria', 'Victoria')], max_length=20)),
                ('created', models.DateTimeField(auto_now_add=True)),
                ('id', models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False, unique=True)),
                ('user', models.OneToOneField(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
        ),
    ]